# Copyright 2020 The TensorFlow Probability Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Core logic for the inverse transformation."""
from typing import Iterable

import jax
from jax import abstract_arrays
from jax import core as jax_core
from jax import linear_util as lu
from jax import tree_util
from jax import util as jax_util
from jax.interpreters import partial_eval as pe
from jax.interpreters import pxla
from jax.interpreters import xla
import jax.numpy as np

from oryx.core import primitive
from oryx.core import trace_util
from oryx.core.interpreters import harvest
from oryx.core.interpreters import propagate
from oryx.core.interpreters.inverse import slice as slc

safe_map = jax_core.safe_map
safe_zip = jax_core.safe_zip
Cell = propagate.Cell
NDSlice = slc.NDSlice
Slice = slc.Slice


class InverseAndILDJ(Cell):
  """Propagates inverse value slices and their ILDJs.

  An InverseAndILDJ instance keeps track of a set of slices of a value. In the
  simplest case, the slice's indices capture the entire value, in which case the
  cell is "top". Partial information is represented with slices that do not
  capture the entire value. No information, i.e. "bottom', is represented with a
  cell that has no slices.

  Joining two cells creates set of slices, and if we detect that the slices can
  be concatenated, we combine them into a single slice. As propagation
  progresses, we hope to accumulate enough slices to concatenate them all into
  this cell's `val`. ILDJs are also kept track of in the same way, except we
  keep track of the diagonal of the Jacobian since split operations may also
  split up the Jacobian.
  """

  def __init__(self,
               aval: jax_core.AbstractValue,
               slices: Iterable[NDSlice]):
    super().__init__(aval)
    self.slices = frozenset(slices)

  def top(self) -> bool:
    """Returns if this cell represents the top of the slice lattice.

    An InverseAndILDJ is at the top if its slice represents the entire array.
    """
    if len(self.slices) != 1:
      return False
    if self.aval == jax_core.abstract_unit:
      return True
    return list(self.slices)[0].value.shape == self.aval.shape

  def bottom(self) -> bool:
    """Returns if this cell represents the bottom of the slice lattice.

    An InverseAndILDJ is at the bottom if we have no slices.
    """
    return len(self.slices) == 0  # pylint: disable=g-explicit-length-test

  def __lt__(self, other: 'InverseAndILDJ') -> bool:
    if self.top() or other.bottom():
      return False
    return all(any(s1 < s2 for s2 in other.slices) for s1 in self.slices)

  def __eq__(self, other: 'InverseAndILDJ') -> bool:
    if self.aval != other.aval:
      return False
    return self.slices == other.slices

  def join(self, other: 'InverseAndILDJ') -> 'InverseAndILDJ':
    if other.top():
      return other
    if other.bottom():
      return self
    if self == other:
      return self
    if other < self:
      return self
    if self < other:
      return other
    all_slices = sorted(self.slices | other.slices,
                        key=lambda slc: tuple(s.start for s in slc.slices))
    new_slices = set()
    active = all_slices.pop(0)
    while all_slices:
      for dim in range(len(self.aval.shape)):
        if active.can_concatenate(all_slices[0], dim):
          active = active.concatenate(all_slices.pop(0), dim)
          break
      else:
        new_slices.add(active)
        active = all_slices.pop(0)
    new_slices.add(active)
    return InverseAndILDJ(self.aval, new_slices)

  @property
  def val(self):
    if not self.top():
      raise AssertionError('Cannot get value from non-top lattice value: ',
                           f'{self.aval}, {self.slices}')
    return list(self.slices)[0].value

  @property
  def ildj(self):
    if not self.top():
      raise AssertionError('Cannot get ildj from non-top lattice value: ',
                           f'{self.aval}, {self.slices}')
    return list(self.slices)[0].ildj

  @classmethod
  def unknown(cls, aval):
    return InverseAndILDJ(aval, [])

  @classmethod
  def new(cls, val):
    val = np.array(val)
    aval = jax_core.get_aval(val)
    if aval is jax_core.abstract_unit:
      return InverseAndILDJ.unknown(aval)
    aval = abstract_arrays.raise_to_shaped(aval)
    ndslice = NDSlice.new(val, np.zeros_like(val))
    return InverseAndILDJ(aval, frozenset([ndslice]))

  def flatten(self):
    slices = list(sorted(self.slices))
    return slices, (self.aval,)

  @classmethod
  def unflatten(cls, data, slices):
    return InverseAndILDJ(data[0], frozenset(slices))


def inverse_and_ildj(f, *trace_args):
  """Inverse and ILDJ function transformation."""
  def wrapped(*args, **kwargs):
    """Function wrapper that takes in inverse arguments."""
    forward_args = trace_args if len(trace_args) else args
    jaxpr, (in_tree, _) = trace_util.stage(f)(*forward_args, **kwargs)
    flat_forward_args, _ = tree_util.tree_flatten(forward_args)
    flat_args, _ = tree_util.tree_flatten(args)
    flat_constcells = safe_map(InverseAndILDJ.new, jaxpr.literals)
    flat_forward_avals = [
        trace_util.get_shaped_aval(arg)
        for arg in flat_forward_args]
    flat_incells = [InverseAndILDJ.unknown(aval) for aval in flat_forward_avals]
    flat_outcells = safe_map(InverseAndILDJ.new, flat_args)
    env = propagate.propagate(InverseAndILDJ, ildj_registry, jaxpr.jaxpr,
                              flat_constcells, flat_incells, flat_outcells)
    flat_incells = [env.read(invar) for invar in jaxpr.jaxpr.invars]
    if any(flat_incell.is_unknown() for flat_incell in flat_incells):
      raise ValueError('Cannot invert function.')
    flat_cells, flat_ildjs = jax_util.unzip2([
        (flat_incell.val, flat_incell.ildj) for flat_incell in flat_incells
    ])
    vals = tree_util.tree_unflatten(in_tree, flat_cells)
    ildjs = tree_util.tree_unflatten(in_tree, flat_ildjs)
    ildj_ = sum(np.sum(i) for i in ildjs)
    if len(forward_args) == 1:
      vals = vals[0]
    return vals, ildj_
  return wrapped


def inverse(f, *trace_args):
  def wrapped(*args, **kwargs):
    return inverse_and_ildj(f, *trace_args)(*args, **kwargs)[0]
  return wrapped


def ildj(f, *trace_args):
  def wrapped(*args, **kwargs):
    return inverse_and_ildj(f, *trace_args)(*args, **kwargs)[1]
  return wrapped


def default_rule(prim, invals, outvals, **params):
  """Default inversion rule that only does forward eval."""
  if all(outval.bottom() for outval in outvals):
    if all(inval.top() for inval in invals):
      vals = [inval.val for inval in invals]
      ans = prim.bind(*vals, **params)
      if not prim.multiple_results:
        ans = [ans]
      # Propagate can only invert functions that are constructed
      # autoregressively, and therefore the Jacobians of propagate-invertible
      # functions are lower-triangular. We are therefore safe assign outvals an
      # ILDJ value of 0 as they are part of forward propagation that will fill
      # in an off-diagonal entry of the Jacobian and will not contribute to the
      # log-det Jacobian.
      outvals = safe_map(InverseAndILDJ.new, ans)
    return invals, outvals, None
  if any(outval.bottom() for outval in outvals):
    return invals, outvals, None
  raise NotImplementedError(f'No registered inverse for `{prim}`.')


class InverseDict(object):
  """Default rules dictionary that uses a default rule for inverse."""

  def __init__(self):
    self.rules = {}

  def __getitem__(self, prim):
    if prim not in self.rules:
      self[prim] = jax_util.partial(default_rule, prim)
    return self.rules[prim]

  def __setitem__(self, prim, val):
    self.rules[prim] = val


def register_elementwise(prim):
  """Registers an elementwise primitive with ILDJ."""
  def make_rule(f):
    """Accepts an inverse function for a primitive."""
    def ildj_rule(incells, outcells, **params):
      """General InverseAndILDJ rule for elementwise functions."""
      outcell, = outcells
      incell, = incells
      if incell.is_unknown() and not outcell.is_unknown():
        val = outcell.val
        f_sum = lambda x: f(x).sum()
        ildj_ = outcell.ildj + np.log(jax.grad(f_sum)(val))
        ndslice = NDSlice.new(f(val), ildj_)
        incells = [InverseAndILDJ(outcell.aval, [ndslice])]
      elif outcell.is_unknown() and not incell.is_unknown():
        outcells = [InverseAndILDJ.new(prim.bind(incell.val, **params))]
      return incells, outcells, None
    ildj_registry[prim] = ildj_rule
  return make_rule


def register_binary(prim):
  """Registers an binary primitive with ILDJ."""
  def make_rule(f_left, f_right):
    def ildj_rule(incells, outcells, **params):
      outcell, = outcells
      left, right = incells
      if not outcell.bottom():
        val, ildj_ = outcell.val, outcell.ildj
        if not left.bottom():
          right_val, right_ildj = f_left(left.val, val, ildj_)
          ndslice = NDSlice.new(right_val, right_ildj)
          incells = [left, InverseAndILDJ(right.aval, [ndslice])]
        elif not right.bottom():
          left_val, left_ildj = f_right(right.val, val, ildj_)
          ndslice = NDSlice.new(left_val, left_ildj)
          incells = [InverseAndILDJ(left.aval, [ndslice]), right]
      elif (outcell.bottom() and not left.bottom() and
            not right.bottom()):
        out_val = prim.bind(left.val, right.val, **params)
        outcells = [InverseAndILDJ.new(out_val)]
      return incells, outcells, None
    ildj_registry[prim] = ildj_rule
  return make_rule


ildj_registry = InverseDict()


@lu.transformation_with_aux
def flat_propagate(tree, *flat_invals):
  invals, outvals = tree_util.tree_unflatten(tree, flat_invals)
  subenv = yield ((invals, outvals), {})
  subenv_vals, subenv_tree = tree_util.tree_flatten(subenv)
  yield subenv_vals, subenv_tree


def call_ildj(prim, incells, outcells, **params):
  """InverseAndILDJ rule for call primitives."""
  f, incells = incells[0], incells[1:]
  flat_vals, in_tree = tree_util.tree_flatten((incells, outcells))
  new_params = dict(params)
  if 'donated_invars' in params:
    new_params['donated_invars'] = (False,) * len(flat_vals)
  f, aux = flat_propagate(f, in_tree)
  subenv_vals = prim.bind(f, *flat_vals, **new_params)
  subenv_tree = aux()
  subenv = tree_util.tree_unflatten(subenv_tree, subenv_vals)
  new_incells = [subenv.read(var) for var in subenv.jaxpr.invars]
  new_outcells = [subenv.read(var) for var in subenv.jaxpr.outvars]
  return new_incells, new_outcells, subenv
ildj_registry[xla.xla_call_p] = jax_util.partial(call_ildj, xla.xla_call_p)
ildj_registry[jax_core.call_p] = jax_util.partial(call_ildj, jax_core.call_p)
ildj_registry[pe.remat_call_p] = jax_util.partial(call_ildj, pe.remat_call_p)
ildj_registry[harvest.nest_p] = jax_util.partial(call_ildj, harvest.nest_p)


def hop_inverse_rule(prim):
  ildj_registry[prim] = jax_util.partial(call_ildj, prim)
primitive.register_hop_transformation_rule('inverse', hop_inverse_rule)


def map_ildj(prim, incells, outcells, **params):
  """InverseAndILDJ rule for the map primitives."""
  f, incells = incells[0], incells[1:]

  def slice_aval(aval):
    return abstract_arrays.ShapedArray(aval.shape[1:], aval.dtype,
                                       aval.weak_type)

  def add_slice(cell, old_cell):
    new_slices = [
        NDSlice(ndslice.value, ndslice.ildj, Slice(0, old_cell.aval.shape[0]),
                *ndslice.slices) for ndslice in cell.slices
    ]
    return InverseAndILDJ(old_cell.aval, new_slices)

  def remove_slice(cell):
    new_slices = [
        NDSlice(ndslice.value, ndslice.ildj, *ndslice.slices[1:])
        for ndslice in cell.slices
    ]
    aval = slice_aval(cell.aval)
    return InverseAndILDJ(aval, new_slices)

  mapped_incells = safe_map(remove_slice, incells)
  mapped_outcells = safe_map(remove_slice, outcells)
  flat_vals, in_tree = tree_util.tree_flatten((mapped_incells, mapped_outcells))
  f, aux = flat_propagate(f, in_tree)
  # Assume all invars as mapped
  new_mapped_invars = (True,) * len(flat_vals)
  new_params = dict(params, mapped_invars=new_mapped_invars)
  subenv_vals = prim.bind(f, *flat_vals, **new_params)
  subenv_tree = aux()
  subenv = tree_util.tree_unflatten(subenv_tree, subenv_vals)
  new_incells = [subenv.read(var) for var in subenv.jaxpr.invars]
  new_outcells = [subenv.read(var) for var in subenv.jaxpr.outvars]
  new_incells = [add_slice(v, old_v)
                 for old_v, v in safe_zip(incells, new_incells)]
  new_outcells = [add_slice(v, old_v)
                  for old_v, v in safe_zip(outcells, new_outcells)]
  return new_incells, new_outcells, subenv
ildj_registry[pxla.xla_pmap_p] = jax_util.partial(map_ildj, pxla.xla_pmap_p)
