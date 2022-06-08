# coding=utf-8
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

from jax.numpy import lax_numpy as jnp
from jax.numpy.vectorize import vectorize
from jax import ad_util
from jax import api
from jax import lax
from jax import ops
from jax import dtypes
from jax.interpreters import xla
from jax.interpreters import ad
from jax.interpreters import batching
from jax.util import partial, prod
from jax.abstract_arrays import ShapedArray
from jax.core import Primitive
from jax.lax import (standard_primitive, standard_unop, naryop_dtype_rule,
                     _float, _complex, _input_dtype, _broadcasting_select)
from jax.lib import lapack
from jax.lib import cusolver

from jax.lib import xla_client
from jax.lib import xla_bridge as xb

xops = xla_client.ops


# traceables

def cholesky(x, symmetrize_input=True):
  if symmetrize_input:
    x = symmetrize(x)
  return jnp.tril(cholesky_p.bind(x))

def eig(x):
  w, vl, vr = eig_p.bind(x)
  return w, vl, vr

def eigh(x, lower=True, symmetrize_input=True):
  if symmetrize_input:
    x = symmetrize(x)
  v, w = eigh_p.bind(x, lower=lower)
  return v, w

def lu(x):
  lu, pivots = lu_p.bind(x)
  return lu, pivots

def qr(x, full_matrices=True):
  q, r = qr_p.bind(x, full_matrices=full_matrices)
  return q, r

def svd(x, full_matrices=True, compute_uv=True):
  s, u, v = svd_p.bind(x, full_matrices=full_matrices, compute_uv=compute_uv)
  if compute_uv:
    return u, s, v
  else:
    return s

def triangular_solve(a, b, left_side=False, lower=False, transpose_a=False,
                     conjugate_a=False, unit_diagonal=False):
  conjugate_a = conjugate_a and jnp.issubdtype(lax.dtype(a), jnp.complexfloating)
  singleton = jnp.ndim(b) == jnp.ndim(a) - 1
  if singleton:
    b = jnp.expand_dims(b, -1 if left_side else -2)
  out = triangular_solve_p.bind(
      a, b, left_side=left_side, lower=lower, transpose_a=transpose_a,
      conjugate_a=conjugate_a, unit_diagonal=unit_diagonal)
  if singleton:
    out = out[..., 0] if left_side else out[..., 0, :]
  return out


# utilities

def _T(x): return jnp.swapaxes(x, -1, -2)
def _H(x): return jnp.conj(_T(x))
def symmetrize(x): return (x + _H(x)) / 2

def _unpack_tuple(f, n):
  def g(c, *args, **kwargs):
    t = f(c, *args, **kwargs)
    return (xops.GetTupleElement(t, i) for i in range(n))
  return g

# primitives

_cpu_lapack_types = {np.dtype(np.float32), np.dtype(np.float64),
                     np.dtype(np.complex64), np.dtype(np.complex128)}

# Cholesky decomposition

def cholesky_jvp_rule(primals, tangents):
  x, = primals
  sigma_dot, = tangents
  L = jnp.tril(cholesky_p.bind(x))

  # Forward-mode rule from https://arxiv.org/pdf/1602.07527.pdf
  def phi(X):
    l = jnp.tril(X)
    return l / (jnp._constant_like(X, 1) + jnp.eye(X.shape[-1], dtype=X.dtype))

  tmp = triangular_solve(L, sigma_dot, left_side=False, transpose_a=True,
                         conjugate_a=True, lower=True)
  L_dot = lax.batch_matmul(L, phi(triangular_solve(
      L, tmp, left_side=True, transpose_a=False, lower=True)),
      precision=lax.Precision.HIGHEST)
  return L, L_dot

def cholesky_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return cholesky(x), 0

cholesky_p = standard_unop(_float | _complex, 'cholesky')
ad.primitive_jvps[cholesky_p] = cholesky_jvp_rule
batching.primitive_batchers[cholesky_p] = cholesky_batching_rule

def _nan_like(c, operand):
  shape = c.get_shape(operand)
  dtype = shape.element_type()
  if jnp.issubdtype(dtype, np.complexfloating):
    nan = xb.constant(c, np.array(np.nan * (1. + 1j), dtype=dtype))
  else:
    nan = xb.constant(c, np.array(np.nan, dtype=dtype))
  return xops.Broadcast(nan, shape.dimensions())

def _cholesky_cpu_gpu_translation_rule(potrf_impl, c, operand):
  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]
  result, info = potrf_impl(c, operand, lower=True)
  ok = xops.Eq(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  return _broadcasting_select(c,
                              xops.Reshape(ok, batch_dims + (1, 1)), result,
                              _nan_like(c, result))

xla.backend_specific_translations['cpu'][cholesky_p] = partial(
  _cholesky_cpu_gpu_translation_rule, lapack.potrf)

xla.backend_specific_translations['gpu'][cholesky_p] = partial(
  _cholesky_cpu_gpu_translation_rule, cusolver.potrf)

# Asymmetric eigendecomposition

def eig_impl(operand):
  return xla.apply_primitive(eig_p, operand)

def eig_translation_rule(c, operand):
  raise NotImplementedError(
    "Nonsymmetric eigendecomposition is only implemented on the CPU backend")

def eig_abstract_eval(operand):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError("Argument to nonsymmetric eigendecomposition must have "
                       "shape [..., n, n], got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    dtype = np.complex64 if dtypes.finfo(operand.dtype).bits == 32 else np.complex128
    dtype = dtypes.canonicalize_dtype(dtype)
    vl = vr = ShapedArray(batch_dims + (n, n), dtype)
    w = ShapedArray(batch_dims + (n,), dtype)
  else:
    raise NotImplementedError
  return w, vl, vr

_cpu_geev = lapack.geev

def eig_cpu_translation_rule(c, operand):
  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]
  w, vl, vr, info = _cpu_geev(c, operand)
  ok = xops.Eq(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  w = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1,)), w,
                           _nan_like(c, w))
  vl = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), vl,
                            _nan_like(c, vl))
  vr = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), vr,
                            _nan_like(c, vr))
  return xops.Tuple(c, [w, vl, vr])

def eig_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return eig_p.bind(x), (0, 0, 0)

eig_p = Primitive('eig')
eig_p.multiple_results = True
eig_p.def_impl(eig_impl)
eig_p.def_abstract_eval(eig_abstract_eval)
xla.translations[eig_p] = eig_translation_rule
xla.backend_specific_translations['cpu'][eig_p] = eig_cpu_translation_rule
batching.primitive_batchers[eig_p] = eig_batching_rule


# Symmetric/Hermitian eigendecomposition

def eigh_impl(operand, lower):
  v, w = xla.apply_primitive(eigh_p, operand, lower=lower)
  return v, w

def eigh_translation_rule(c, operand, lower):
  shape = c.get_shape(operand)
  dims = shape.dimensions()
  if dims[-1] == 0:
    return xops.Tuple(c, [operand, xops.Reshape(operand, dims[:-1])])
  if not lower:
    n = len(dims)
    operand = xops.Transpose(operand, list(range(n - 2)) + [n - 1, n - 2])
  return xops.Tuple(c, xops.Eigh(operand))

def eigh_abstract_eval(operand, lower):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2 or operand.shape[-2] != operand.shape[-1]:
      raise ValueError(
        "Argument to symmetric eigendecomposition must have shape [..., n, n],"
        "got shape {}".format(operand.shape))

    batch_dims = operand.shape[:-2]
    n = operand.shape[-1]
    v = ShapedArray(batch_dims + (n, n), operand.dtype)
    w = ShapedArray(batch_dims + (n,), lax.lax._complex_basetype(operand.dtype))
  else:
    v, w = operand, operand
  return v, w

def _eigh_cpu_gpu_translation_rule(syevd_impl, c, operand, lower):
  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]
  v, w, info = syevd_impl(c, operand, lower=lower)
  ok = xops.Eq(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  v = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), v,
                           _nan_like(c, v))
  w = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1,)), w,
                           _nan_like(c, w))
  return xops.Tuple(c, [v, w])

def eigh_jvp_rule(primals, tangents, lower):
  # Derivative for eigh in the simplest case of distinct eigenvalues.
  # This is classic nondegenerate perurbation theory, but also see
  # https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
  # The general solution treating the case of degenerate eigenvalues is
  # considerably more complicated. Ambitious readers may refer to the general
  # methods below or refer to degenerate perturbation theory in physics.
  # https://www.win.tue.nl/analysis/reports/rana06-33.pdf and
  # https://people.orie.cornell.edu/aslewis/publications/99-clarke.pdf
  a, = primals
  a_dot, = tangents

  v, w_real = eigh_p.bind(symmetrize(a), lower=lower)

  # for complex numbers we need eigenvalues to be full dtype of v, a:
  w = w_real.astype(a.dtype)
  eye_n = jnp.eye(a.shape[-1], dtype=a.dtype)
  # carefully build reciprocal delta-eigenvalue matrix, avoiding NaNs.
  Fmat = jnp.reciprocal(eye_n + w[..., jnp.newaxis, :] - w[..., jnp.newaxis]) - eye_n
  # eigh impl doesn't support batch dims, but future-proof the grad.
  dot = partial(lax.dot if a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)
  vdag_adot_v = dot(dot(_H(v), a_dot), v)
  dv = dot(v, jnp.multiply(Fmat, vdag_adot_v))
  dw = jnp.real(jnp.diagonal(vdag_adot_v, axis1=-2, axis2=-1))
  return (v, w_real), (dv, dw)

def eigh_batching_rule(batched_args, batch_dims, lower):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return eigh_p.bind(x, lower=lower), (0, 0)

eigh_p = Primitive('eigh')
eigh_p.multiple_results = True
eigh_p.def_impl(eigh_impl)
eigh_p.def_abstract_eval(eigh_abstract_eval)
xla.translations[eigh_p] = eigh_translation_rule
ad.primitive_jvps[eigh_p] = eigh_jvp_rule
batching.primitive_batchers[eigh_p] = eigh_batching_rule

_cpu_syevd = lapack.syevd

xla.backend_specific_translations['cpu'][eigh_p] = partial(
  _eigh_cpu_gpu_translation_rule, _cpu_syevd)

xla.backend_specific_translations['gpu'][eigh_p] = partial(
  _eigh_cpu_gpu_translation_rule, cusolver.syevd)




triangular_solve_dtype_rule = partial(
    naryop_dtype_rule, _input_dtype, (_float | _complex, _float | _complex),
    'triangular_solve')

def triangular_solve_shape_rule(a, b, left_side=False, **unused_kwargs):
  if a.ndim < 2:
    msg = "triangular_solve requires a.ndim to be at least 2, got {}."
    raise TypeError(msg.format(a.ndim))
  if b.ndim < 2:
    msg = "triangular_solve requires b.ndim to be at least 2, got {}."
    raise TypeError(msg.format(b.ndim))
  if a.shape[-1] != a.shape[-2]:
    msg = ("triangular_solve requires the last two dimensions of a to be equal "
           "in size, got a.shape of {}.")
    raise TypeError(msg.format(a.shape))
  if a.shape[:-2] != b.shape[:-2]:
    msg = ("triangular_solve requires both arguments to have the same number "
           "of dimensions and equal batch dimensions, got {} and {}.")
    raise TypeError(msg.format(a.shape, b.shape))
  common_dim = -2 if left_side else -1
  if a.shape[-1] != b.shape[common_dim]:
    msg = "Incompatible shapes for arguments to triangular_solve: {} and {}."
    raise TypeError(msg.format(a.shape, b.shape))
  return b.shape

def triangular_solve_jvp_rule_a(
    g_a, ans, a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  m, n = b.shape[-2:]
  k = 1 if unit_diagonal else 0
  g_a = jnp.tril(g_a, k=-k) if lower else jnp.triu(g_a, k=k)
  g_a = lax.neg(g_a)
  g_a = jnp.swapaxes(g_a, -1, -2) if transpose_a else g_a
  g_a = jnp.conj(g_a) if conjugate_a else g_a
  dot = partial(lax.dot if g_a.ndim == 2 else lax.batch_matmul,
                precision=lax.Precision.HIGHEST)

  def a_inverse(rhs):
    return triangular_solve(a, rhs, left_side, lower, transpose_a, conjugate_a,
                            unit_diagonal)

  # triangular_solve is about the same cost as matrix multplication (~n^2 FLOPs
  # for matrix/vector inputs). Order these operations in whichever order is
  # cheaper.
  if left_side:
    assert g_a.shape[-2:] == a.shape[-2:] == (m, m) and ans.shape[-2:] == (m, n)
    if m > n:
      return a_inverse(dot(g_a, ans))  # A^{-1} (∂A X)
    else:
      return dot(a_inverse(g_a), ans)  # (A^{-1} ∂A) X
  else:
    assert g_a.shape[-2:] == a.shape[-2:] == (n, n) and ans.shape[-2:] == (m, n)
    if m < n:
      return a_inverse(dot(ans, g_a))  # (X ∂A) A^{-1}
    else:
      return dot(ans, a_inverse(g_a))  # X (∂A A^{-1})

def triangular_solve_transpose_rule(
    cotangent, a, b, left_side, lower, transpose_a, conjugate_a,
    unit_diagonal):
  # Triangular solve is nonlinear in its first argument and linear in its second
  # argument, analogous to `div` but swapped.
  assert not ad.is_undefined_primal(a) and ad.is_undefined_primal(b)
  if type(cotangent) is ad_util.Zero:
    cotangent_b = ad_util.Zero(b.aval)
  else:
    cotangent_b = triangular_solve(a, cotangent, left_side, lower,
                                   not transpose_a, conjugate_a, unit_diagonal)
  return [None, cotangent_b]


def triangular_solve_batching_rule(batched_args, batch_dims, left_side,
                                   lower, transpose_a, conjugate_a,
                                   unit_diagonal):
  x, y = batched_args
  bx, by = batch_dims
  if bx is batching.not_mapped:
    if left_side:
      y = batching.moveaxis(y, by, -1)
      y_flat = y.reshape(y.shape[:-2] + (y.shape[-2] * y.shape[-1],))
      bdim_out = y.ndim - 1
    else:
      y = batching.moveaxis(y, by, -2)
      y_flat = y.reshape(y.shape[:-3]  + (y.shape[-3] * y.shape[-2], y.shape[-1]))
      bdim_out = y.ndim - 2
    out_flat = triangular_solve(
        x, y_flat, left_side=left_side, lower=lower,
        transpose_a=transpose_a, conjugate_a=conjugate_a,
        unit_diagonal=unit_diagonal)
    return out_flat.reshape(y.shape), bdim_out
  else:
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims)
                if i is not None)
    x = batching.bdim_at_front(x, bx, size)
    y = batching.bdim_at_front(y, by, size)
    return triangular_solve(x, y, left_side=left_side, lower=lower,
                            transpose_a=transpose_a, conjugate_a=conjugate_a,
                            unit_diagonal=unit_diagonal), 0

def _triangular_solve_translation_rule(
    c, a, b, *, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  if conjugate_a and not transpose_a:
    a = xops.Conj(a)
    conjugate_a = False
  if not transpose_a:
    transpose = xops.TriangularSolveOptions_Transpose.NO_TRANSPOSE
  else:
    transpose = (xops.TriangularSolveOptions_Transpose.ADJOINT if conjugate_a
                 else xops.TriangularSolveOptions_Transpose.TRANSPOSE)
  return xops.TriangularSolve(a, b, left_side, lower, unit_diagonal, transpose)

triangular_solve_p = standard_primitive(
    triangular_solve_shape_rule, triangular_solve_dtype_rule,
    'triangular_solve', translation_rule=_triangular_solve_translation_rule)
ad.defjvp2(triangular_solve_p,
           triangular_solve_jvp_rule_a,
           lambda g_b, _, a, b, **kws: triangular_solve(a, g_b, **kws))
ad.primitive_transposes[triangular_solve_p] = triangular_solve_transpose_rule
batching.primitive_batchers[triangular_solve_p] = triangular_solve_batching_rule


def _triangular_solve_cpu_translation_rule(
    c, a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  shape = c.get_shape(a)
  dtype = shape.element_type().type

  if conjugate_a and not transpose_a:
    a = xops.Conj(a)
    conjugate_a = False
  if len(shape.dimensions()) == 2 and np.dtype(dtype) in _cpu_lapack_types:
    return lapack.jax_trsm(
      c, xb.constant(c, np.array(1, dtype=dtype)),
      a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal)
  else:
    # Fall back to the HLO implementation for unsupported types or batching.
    # TODO: Consider swapping XLA for LAPACK in batched case
    if not transpose_a:
      transpose = xops.TriangularSolveOptions_Transpose.NO_TRANSPOSE
    else:
      transpose = (xops.TriangularSolveOptions_Transpose.ADJOINT if conjugate_a
                   else xops.TriangularSolveOptions_Transpose.TRANSPOSE)
    return xops.TriangularSolve(a, b, left_side, lower, unit_diagonal, transpose)

xla.backend_specific_translations['cpu'][triangular_solve_p] = \
  _triangular_solve_cpu_translation_rule

def _triangular_solve_gpu_translation_rule(
    c, a, b, left_side, lower, transpose_a, conjugate_a, unit_diagonal):
  shape = c.get_shape(a)
  dims = shape.dimensions()
  m, n = dims[-2:]
  batch = prod(dims[:-2])
  if conjugate_a and not transpose_a:
    a = xops.Conj(a)
    conjugate_a = False
  if batch > 1 and m <= 32 and n <= 32:
    return cusolver.trsm(
      c, a, b, left_side, lower, transpose_a,
      conjugate_a, unit_diagonal)
  else:
    # Use the XLA implementation for unbatched triangular_solve.
    if not transpose_a:
      transpose = xops.TriangularSolveOptions_Transpose.NO_TRANSPOSE
    else:
      transpose = (xops.TriangularSolveOptions_Transpose.ADJOINT if conjugate_a
                   else xops.TriangularSolveOptions_Transpose.TRANSPOSE)
    return xops.TriangularSolve(a, b, left_side, lower, unit_diagonal,
                                transpose)

xla.backend_specific_translations['gpu'][triangular_solve_p] = \
    _triangular_solve_gpu_translation_rule

# LU decomposition

# Computes a pivoted LU decomposition such that
# PA = LU
# In the style of LAPACK, LU are stored in the same matrix.

def _lu_unblocked(a):
  """Unblocked LU decomposition, as a rolled loop."""
  m, n = a.shape
  def body(k, state):
    pivot, perm, a = state
    m_idx = jnp.arange(m)
    n_idx = jnp.arange(n)

    if jnp.issubdtype(a.dtype, jnp.complexfloating):
      t = a[:, k]
      magnitude = jnp.abs(jnp.real(t)) + jnp.abs(jnp.imag(t))
    else:
      magnitude = jnp.abs(a[:, k])
    i = jnp.argmax(jnp.where(m_idx >= k, magnitude, -jnp.inf))
    pivot = ops.index_update(pivot, ops.index[k], i)

    a = ops.index_update(a, ops.index[[k, i],], a[[i, k],])

    perm = ops.index_update(perm, ops.index[[i, k],], perm[[k, i],])

    # a[k+1:, k] /= a[k, k], adapted for loop-invariant shapes
    x = a[k, k]
    a = ops.index_update(a, ops.index[:, k],
                         jnp.where(m_idx > k, a[:, k] / x, a[:, k]))

    # a[k+1:, k+1:] -= jnp.outer(a[k+1:, k], a[k, k+1:])
    a = a - jnp.where((m_idx[:, None] > k) & (n_idx > k),
                     jnp.outer(a[:, k], a[k, :]), jnp.array(0, dtype=a.dtype))
    return pivot, perm, a

  pivot = jnp.zeros((min(m, n),), dtype=jnp.int32)
  perm = jnp.arange(m, dtype=jnp.int32)
  if m == 0 and n == 0:
    # If the array is empty, the loop body never executes but tracing it to a
    # jaxpr fails because the indexing cannot succeed.
    return (pivot, perm, a)
  return lax.fori_loop(0, min(m, n), body, (pivot, perm, a))


def _lu_blocked(a, block_size=128):
  """Blocked LU decomposition, as an unrolled loop."""
  m, n = a.shape
  r = min(m, n)
  pivot = jnp.zeros((r,), dtype=jnp.int32)
  for k in range(0, r, block_size):
    b = min(r - k, block_size)
    block_pivot, perm, lu_block = _lu_unblocked(a[k:, k:k+b])

    a = ops.index_update(a, ops.index[k:, :], a[perm + k, :])
    a = ops.index_update(a, ops.index[k:, k:k+b], lu_block)
    pivot = ops.index_update(pivot, ops.index[k:k+b], block_pivot + k)

    if k + b < n:
      a = ops.index_update(
        a, ops.index[k:k+b, k+b:],
        triangular_solve(a[k:k+b, k:k+b], a[k:k+b, k+b:],
                         left_side=True, lower=True, unit_diagonal=True))
      a = ops.index_add(
        a, ops.index[k+b:, k+b:],
        -lax.dot(a[k+b:, k:k+b], a[k:k+b, k+b:],
                 precision=lax.Precision.HIGHEST))
  return pivot, a

def _lu_python(x):
  """Default LU decomposition in Python, where no better version exists."""
  m, n = x.shape[-2:]
  batch_dims = x.shape[:-2]
  if len(batch_dims) > 0:
    batch_size = np.prod(batch_dims, dtype=np.int64)
    pivot, lu = api.vmap(_lu_blocked)(lax.reshape(x, (batch_size, m, n)))
    pivot = lax.reshape(pivot, batch_dims + (min(m, n),))
    lu = lax.reshape(lu, batch_dims + (m, n))
  else:
    pivot, lu = _lu_blocked(x)
  return lu, pivot

def _lu_impl(operand):
  lu, pivot = xla.apply_primitive(lu_p, operand)
  return lu, pivot

def _lu_abstract_eval(operand):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to LU decomposition must have ndims >= 2")

    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    pivot = ShapedArray(batch_dims + (min(m, n),), jnp.int32)
  else:
    pivot = operand
  return operand, pivot

def _lu_jvp_rule(primals, tangents):
  a, = primals
  a_dot, = tangents
  lu, pivots = lu_p.bind(a)

  a_shape = jnp.shape(a)
  m, n = a_shape[-2:]
  dtype = lax.dtype(a)
  k = min(m, n)

  permutation = lu_pivots_to_permutation(pivots, m)
  batch_dims = a_shape[:-2]
  iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in batch_dims + (1,)))
  x = a_dot[iotas[:-1] + (permutation, slice(None))]

  # Differentiation of Matrix Functionals Using Triangular Factorization
  # F. R. De Hoog, R. S. Anderssen, and M. A. Lukas
  #
  #     LU = A
  # ==> L'U + LU' = A'
  # ==> inv(L) . L' + U' . inv(U) = inv(L) A' inv(U)
  # ==> L' = L . tril(inv(L) . A' . inv(U), -1)
  #     U' = triu(inv(L) . A' . inv(U)) . U

  ndims = len(a_shape)
  l_padding = [(0, 0, 0)] * ndims
  l_padding[-1] = (0, m - k, 0)
  zero = jnp._constant_like(lu, 0)
  l = lax.pad(jnp.tril(lu[..., :, :k], -1), zero, l_padding)
  l = l + jnp.eye(m, m, dtype=dtype)

  u_eye = lax.pad(jnp.eye(n - k, n - k, dtype=dtype), zero,
                  ((k, 0, 0), (k, 0, 0)))
  u_padding = [(0, 0, 0)] * ndims
  u_padding[-2] = (0, n - k, 0)
  u = lax.pad(jnp.triu(lu[..., :k, :]), zero, u_padding) + u_eye

  la = triangular_solve(l, x, left_side=True, transpose_a=False, lower=True,
                        unit_diagonal=True)
  lau = triangular_solve(u, la, left_side=False, transpose_a=False,
                         lower=False)

  l_dot = jnp.matmul(l, jnp.tril(lau, -1))
  u_dot = jnp.matmul(jnp.triu(lau), u)
  lu_dot = l_dot + u_dot
  return (lu, pivots), (lu_dot, ad_util.Zero.from_value(pivots))


def _lu_batching_rule(batched_args, batch_dims):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return lu_p.bind(x), (0, 0)

def _lu_cpu_gpu_translation_rule(getrf_impl, c, operand):
  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]
  lu, pivot, info = getrf_impl(c, operand)
  # Subtract 1 from the pivot to get 0-based indices.
  pivot = xops.Sub(pivot, xops.ConstantLiteral(c, np.array(1, np.int32)))
  ok = xops.Ge(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  lu = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), lu,
                            _nan_like(c, lu))
  return xops.Tuple(c, [lu, pivot])


lu_p = Primitive('lu')
lu_p.multiple_results = True
lu_p.def_impl(_lu_impl)
lu_p.def_abstract_eval(_lu_abstract_eval)
xla.translations[lu_p] = xla.lower_fun(_lu_python, multiple_results=True)
ad.primitive_jvps[lu_p] = _lu_jvp_rule
batching.primitive_batchers[lu_p] = _lu_batching_rule

xla.backend_specific_translations['cpu'][lu_p] = partial(
  _lu_cpu_gpu_translation_rule, lapack.getrf)

xla.backend_specific_translations['gpu'][lu_p] = partial(
  _lu_cpu_gpu_translation_rule, cusolver.getrf)


# Define this outside lu_pivots_to_permutation to ensure fori_loop cache hits
def _lu_pivots_body_fn(i, permutation_and_swaps):
  permutation, swaps = permutation_and_swaps
  batch_dims = swaps.shape[:-1]
  j = swaps[..., i]
  iotas = jnp.ix_(*(lax.iota(jnp.int32, b) for b in batch_dims))
  x = permutation[..., i]
  y = permutation[iotas + (j,)]
  permutation = ops.index_update(permutation, ops.index[..., i], y)
  return ops.index_update(permutation, ops.index[iotas + (j,)], x), swaps


@partial(api.jit, static_argnums=(1,))
def lu_pivots_to_permutation(swaps, m):
  """Converts the pivots (row swaps) returned by LU to a permutation.

  We build a permutation rather than applying `swaps` directly to the rows
  of a matrix because lax loops aren't differentiable.

  Args:
    swaps: an array of shape (..., k) of row swaps to perform
    m: the size of the output permutation. m should be >= k.
  Returns:
    An int32 array of shape (..., m).
  """
  assert len(swaps.shape) >= 1
  batch_dims = swaps.shape[:-1]
  k = swaps.shape[-1]

  permutation = lax.broadcasted_iota(jnp.int32, batch_dims + (m,),
                                     len(batch_dims))
  result, _ = lax.fori_loop(np.array(0, np.int32), np.array(k, np.int32),
                            _lu_pivots_body_fn, (permutation, swaps))
  return result


@partial(vectorize, excluded={3}, signature='(n,n),(n),(n,k)->(n,k)')
def _lu_solve_core(lu, pivots, b, trans):
  m = lu.shape[0]
  permutation = lu_pivots_to_permutation(pivots, m)
  x = jnp.reshape(b, (m, -1))
  if trans == 0:
    x = x[permutation, :]
    x = triangular_solve(lu, x, left_side=True, lower=True, unit_diagonal=True)
    x = triangular_solve(lu, x, left_side=True, lower=False)
  elif trans == 1 or trans == 2:
    conj = trans == 2
    x = triangular_solve(lu, x, left_side=True, lower=False, transpose_a=True,
                         conjugate_a=conj)
    x = triangular_solve(lu, x, left_side=True, lower=True, unit_diagonal=True,
                         transpose_a=True, conjugate_a=conj)
    x = x[jnp.argsort(permutation), :]
  else:
    raise ValueError("'trans' value must be 0, 1, or 2, got {}".format(trans))
  return lax.reshape(x, b.shape)


@partial(api.jit, static_argnums=(3,))
def _lu_solve(lu, pivots, b, trans):
  if len(lu.shape) < 2 or lu.shape[-1] != lu.shape[-2]:
    raise ValueError("last two dimensions of LU decomposition must be equal, "
                     "got shape {}".format(lu.shape))
  if len(b.shape) < 1:
    raise ValueError("b matrix must have rank >= 1, got shape {}"
                     .format(b.shape))
  # Broadcasting follows NumPy's convention for linalg.solve: the RHS is
  # treated as a (batched) vector if the number of dimensions differ by 1.
  # Otherwise, broadcasting rules apply.
  rhs_vector = lu.ndim == b.ndim + 1
  if rhs_vector:
    if b.shape[-1] != lu.shape[-1]:
      raise ValueError("When LU decomposition matrix and b have the same "
                       "number of dimensions, last axis of LU decomposition "
                       "matrix (shape {}) and b array (shape {}) must match"
                       .format(lu.shape, b.shape))
    b = b[..., jnp.newaxis]
  else:
    if b.shape[-2] != lu.shape[-1]:
      raise ValueError("When LU decomposition matrix and b different "
                       "numbers of dimensions, last axis of LU decomposition "
                       "matrix (shape {}) and second to last axis of b array "
                       "(shape {}) must match"
                       .format(lu.shape, b.shape))
  x = _lu_solve_core(lu, pivots, b, trans)
  return x[..., 0] if rhs_vector else x


def lu_solve(lu, pivots, b, trans=0):
  """LU solve with broadcasting."""
  return _lu_solve(lu, pivots, b, trans)


# QR decomposition

def qr_impl(operand, full_matrices):
  q, r = xla.apply_primitive(qr_p, operand, full_matrices=full_matrices)
  return q, r

def qr_translation_rule(c, operand, full_matrices):
  return xops.Tuple(c, xops.QR(operand, full_matrices))

def qr_abstract_eval(operand, full_matrices):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to QR decomposition must have ndims >= 2")
    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    k = m if full_matrices else min(m, n)
    q = ShapedArray(batch_dims + (m, k), operand.dtype)
    r = ShapedArray(batch_dims + (k, n), operand.dtype)
  else:
    q = operand
    r = operand
  return q, r

def qr_jvp_rule(primals, tangents, full_matrices):
  # See j-towns.github.io/papers/qr-derivative.pdf for a terse derivation.
  x, = primals
  dx, = tangents
  q, r = qr_p.bind(x, full_matrices=False)
  *_, m, n = x.shape
  if full_matrices or m < n:
    raise NotImplementedError(
      "Unimplemented case of QR decomposition derivative")
  dx_rinv = triangular_solve(r, dx)  # Right side solve by default
  qt_dx_rinv = jnp.matmul(_H(q), dx_rinv)
  qt_dx_rinv_lower = jnp.tril(qt_dx_rinv, -1)
  do = qt_dx_rinv_lower - _H(qt_dx_rinv_lower)  # This is skew-symmetric
  # The following correction is necessary for complex inputs
  do = do + jnp.eye(n, dtype=do.dtype) * (qt_dx_rinv - jnp.real(qt_dx_rinv))
  dq = jnp.matmul(q, do - qt_dx_rinv) + dx_rinv
  dr = jnp.matmul(qt_dx_rinv - do, r)
  return (q, r), (dq, dr)

def qr_batching_rule(batched_args, batch_dims, full_matrices):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  return qr_p.bind(x, full_matrices=full_matrices), (0, 0)

def _qr_cpu_gpu_translation_rule(geqrf_impl, orgqr_impl, c, operand,
                                 full_matrices):
  shape = c.get_shape(operand)
  dims = shape.dimensions()
  m, n = dims[-2:]
  batch_dims = dims[:-2]
  r, tau, info_geqrf = geqrf_impl(c, operand)
  if m < n:
    q = xops.Slice(r, [0] * len(dims), list(batch_dims) + [m, m],
                   [1] * len(dims))
    q, info_orgqr = orgqr_impl(c, q, tau)
  elif not full_matrices:
    q, info_orgqr = orgqr_impl(c, r, tau)
    r = xops.Slice(r, [0] * len(dims), list(batch_dims) + [n, n],
                   [1] * len(dims))
  else:
    padding_config = [(0, 0, 0)] * len(dims)
    padding_config[-1] = (0, m - n, 0)
    q = xops.Pad(r, xops.Constant(c, np.array(0, dtype=shape.element_type())),
                 xla_client.make_padding_config(padding_config))
    q, info_orgqr = orgqr_impl(c, q, tau)

  ok = xops.And(
    xops.Eq(info_geqrf, xops.ConstantLiteral(c, np.array(0, np.int32))),
    xops.Eq(info_orgqr, xops.ConstantLiteral(c, np.array(0, np.int32))))
  q = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), q,
                           _nan_like(c, q))
  r = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), r,
                           _nan_like(c, r))
  r = xla.lower_fun(jnp.triu, multiple_results=False)(c, r)
  return xops.Tuple(c, [q, r])

qr_p = Primitive('qr')
qr_p.multiple_results = True
qr_p.def_impl(qr_impl)
qr_p.def_abstract_eval(qr_abstract_eval)
xla.translations[qr_p] = qr_translation_rule
ad.primitive_jvps[qr_p] = qr_jvp_rule
batching.primitive_batchers[qr_p] = qr_batching_rule

xla.backend_specific_translations['cpu'][qr_p] = partial(
  _qr_cpu_gpu_translation_rule, lapack.geqrf, lapack.orgqr)

xla.backend_specific_translations['gpu'][qr_p] = partial(
  _qr_cpu_gpu_translation_rule, cusolver.geqrf, cusolver.orgqr)


# Singular value decomposition

def svd_impl(operand, full_matrices, compute_uv):
  s, u, vt = xla.apply_primitive(svd_p, operand, full_matrices=full_matrices,
                                 compute_uv=compute_uv)
  return s, u, vt

def svd_translation_rule(c, operand, full_matrices, compute_uv):
  shape = c.get_shape(operand).dimensions()
  m, n = shape[-2:]
  u, s, v = xops.SVD(operand)
  permutation = list(range(len(shape)))
  permutation[-1], permutation[-2] = permutation[-2], permutation[-1]
  vt = xops.Transpose(v, permutation)
  if not full_matrices and m != n:
    u = xops.SliceInDim(u, 0, min(m, n), stride=1, dimno=len(shape) - 1)
    vt = xops.SliceInDim(vt, 0, min(m, n), stride=1, dimno=len(shape) - 2)
  return xops.Tuple(c, [s, u, vt])

def svd_abstract_eval(operand, full_matrices, compute_uv):
  if isinstance(operand, ShapedArray):
    if operand.ndim < 2:
      raise ValueError("Argument to singular value decomposition must have ndims >= 2")

    batch_dims = operand.shape[:-2]
    m = operand.shape[-2]
    n = operand.shape[-1]
    s = ShapedArray(batch_dims + (min(m, n),), lax.lax._complex_basetype(operand.dtype))
    u = ShapedArray(batch_dims + (m, m if full_matrices else min(m, n)), operand.dtype)
    vt = ShapedArray(batch_dims + (n if full_matrices else min(m, n), n), operand.dtype)
  else:
    raise NotImplementedError
  return s, u, vt

def svd_jvp_rule(primals, tangents, full_matrices, compute_uv):
  A, = primals
  dA, = tangents
  s, U, Vt = svd_p.bind(A, full_matrices=False, compute_uv=True)

  if compute_uv and full_matrices:
    # TODO: implement full matrices case, documented here: https://people.maths.ox.ac.uk/gilesm/files/NA-08-01.pdf
    raise NotImplementedError(
      "Singular value decomposition JVP not implemented for full matrices")

  k = s.shape[-1]
  Ut, V = _H(U), _H(Vt)
  s_dim = s[..., None, :]
  dS = jnp.matmul(jnp.matmul(Ut, dA), V)
  ds = jnp.real(jnp.diagonal(dS, 0, -2, -1))
  F = 1 / (jnp.square(s_dim) - jnp.square(_T(s_dim)) + jnp.eye(k, dtype=A.dtype))
  F = F - jnp.eye(k, dtype=A.dtype)
  dSS = s_dim * dS
  SdS = _T(s_dim) * dS
  dU = jnp.matmul(U, F * (dSS + _T(dSS)))
  dV = jnp.matmul(V, F * (SdS + _T(SdS)))

  m, n = A.shape[-2:]
  if m > n:
    dU = dU + jnp.matmul(jnp.eye(m, dtype=A.dtype) - jnp.matmul(U, Ut), jnp.matmul(dA, V)) / s_dim
  if n > m:
    dV = dV + jnp.matmul(jnp.eye(n, dtype=A.dtype) - jnp.matmul(V, Vt), jnp.matmul(_H(dA), U)) / s_dim
  return (s, U, Vt), (ds, dU, _T(dV))

def _svd_cpu_gpu_translation_rule(gesvd_impl, c, operand, full_matrices, compute_uv):

  shape = c.get_shape(operand)
  batch_dims = shape.dimensions()[:-2]
  s, u, vt, info = gesvd_impl(c, operand,
                              full_matrices=full_matrices,
                              compute_uv=compute_uv)
  ok = xops.Eq(info, xops.ConstantLiteral(c, np.array(0, np.int32)))
  s = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1,)), s,
                           _nan_like(c, s))
  u = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), u,
                           _nan_like(c, u))
  vt = _broadcasting_select(c, xops.Reshape(ok, batch_dims + (1, 1)), vt,
                            _nan_like(c, vt))
  return xops.Tuple(c, [s, u, vt])

def svd_batching_rule(batched_args, batch_dims, full_matrices, compute_uv):
  x, = batched_args
  bd, = batch_dims
  x = batching.moveaxis(x, bd, 0)
  outs = svd_p.bind(x, full_matrices=full_matrices, compute_uv=compute_uv)
  return outs, (0, 0, 0)

svd_p = Primitive('svd')
svd_p.multiple_results = True
svd_p.def_impl(svd_impl)
svd_p.def_abstract_eval(svd_abstract_eval)
ad.primitive_jvps[svd_p] = svd_jvp_rule
batching.primitive_batchers[svd_p] = svd_batching_rule
xla.translations[svd_p] = svd_translation_rule

xla.backend_specific_translations['cpu'][svd_p] = partial(
  _svd_cpu_gpu_translation_rule, lapack.gesdd)

xla.backend_specific_translations['gpu'][svd_p] = partial(
  _svd_cpu_gpu_translation_rule, cusolver.gesvd)
