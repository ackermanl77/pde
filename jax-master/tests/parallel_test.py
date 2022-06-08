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

import functools
import itertools
import unittest
from unittest import SkipTest, skip

import numpy as np
from absl.testing import absltest
from absl.testing import parameterized

import jax.numpy as jnp
from jax import test_util as jtu
from jax import lax
from jax.api import _papply, _parallelize, soft_pmap, jit, make_jaxpr
from jax.util import prod

from jax.config import config
config.parse_flags_with_absl()


ignore_soft_pmap_warning = functools.partial(
  jtu.ignore_warning, message="soft_pmap is an experimental.*")

class PapplyTest(jtu.JaxTestCase):

  def testIdentity(self):
    pfun, axis_name = _papply(lambda x: x)
    ans = pfun(np.arange(3))
    expected = np.arange(3)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testMap(self):
    pfun, axis_name = _papply(jnp.sin)
    ans = pfun(np.arange(3.))
    expected = np.sin(np.arange(3.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSum(self):
    pfun, axis_name = _papply(lambda x: jnp.sum(x, axis=0))

    jaxpr = make_jaxpr(pfun)(np.ones(3))
    expected_jaxpr = make_jaxpr(
        lambda x: lax.psum(x, axis_name))(np.zeros((5, 3)))
    assert repr(jaxpr) == repr(expected_jaxpr)

    arg = np.arange(15.).reshape((5, 3))
    ans = soft_pmap(pfun, axis_name)(arg)[0]
    expected = np.sum(arg, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testMax(self):
    pfun, axis_name = _papply(lambda x: jnp.max(x, axis=0))

    jaxpr = make_jaxpr(pfun)(np.ones(3))
    expected_jaxpr = make_jaxpr(
        lambda x: lax.pmax(x, axis_name))(np.zeros((5, 3)))
    assert repr(jaxpr) == repr(expected_jaxpr)

    arg = np.arange(15.).reshape((5, 3))
    ans = soft_pmap(pfun, axis_name)(arg)[0]
    expected = np.max(arg, axis=0)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testSelect(self):
    p = np.arange(15).reshape((5, 3)) % 4 == 1
    f = np.zeros((5, 3))

    def fun(t):
      return lax.select(p, t, f)

    t = np.ones((5, 3))
    ans = soft_pmap(*_papply(fun))(t)
    expected = fun(t)
    self.assertAllClose(ans, expected)

  def testLogSoftmax(self):
    raise SkipTest("test doesn't pass yet")  # TODO(frostig)

    def fun(x):
      return x - jnp.log(jnp.sum(jnp.exp(x)))

    pfun, axis_name = _papply(fun)

    jaxpr = make_jaxpr(pfun)(np.zeros(5))
    expected_jaxpr = make_jaxpr(
        lambda x: x - jnp.log(lax.psum(jnp.exp(x), axis_name)))(np.zeros(5))
    assert repr(jaxpr) == repr(expected_jaxpr)

    ans = soft_pmap(pfun, axis_name)(np.arange(1., 5.))
    expected = fun(np.arange(1., 5.))
    self.assertAllClose(ans, expected, check_dtypes=False)

  @ignore_soft_pmap_warning()
  def testAdd(self):
    x = np.array([[1, 2, 3], [4, 5, 6]])
    expected = x + x

    pfun, axis_name = _papply(jnp.add)
    ans = soft_pmap(pfun, axis_name)(x, x)
    self.assertAllClose(ans, expected)

  def testAddBroadcasting(self):
    raise SkipTest("test doesn't pass yet")  # TODO(frostig)

    def fun(x):
      return x + 3

    x = np.array([[1, 2], [3, 4]])
    expected = x + 3

    pfun, axis_name = _papply(fun)
    ans = soft_pmap(pfun, axis_name)(x)
    self.assertAllClose(ans, expected)

  def testMakeJaxprPapplyComposition(self):
    raise SkipTest(             # TODO(mattjj)
        "fails because select's papply rule calls an SPMD primitive")
    x = b = np.ones(3)
    pfun, axis_name = _papply(lambda a: jnp.where(x, a, b))
    make_jaxpr(pfun)(np.ones(3))  # doesn't crash


@skip("causing trace state errors that affect other tests")
class ParallelizeTest(jtu.JaxTestCase):

  def dedup(self, arr, expected_rank):
    if arr.ndim == expected_rank + 1:
      for i in range(arr.shape[0] - 1):
        self.assertAllClose(arr[i], arr[i + 1])
      return arr[0]
    else:
      assert arr.ndim == expected_rank
      return arr

  def testNormalize(self):

    def f(x):
      return x / x.sum(0)

    x = np.arange(4.)
    expected = f(x)
    ans = _parallelize(f)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

    jaxpr = make_jaxpr(_parallelize(f))(x)
    self.assertIn('psum', repr(jaxpr))

  def testAdd(self):
    x = np.arange(10)
    y = 2 * np.arange(10)
    def f(x): return x + y
    expected = f(x)
    ans = _parallelize(f)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAdd2(self):
    x = np.arange(10)
    y = 2 * np.arange(10)
    def f(y): return x + y
    expected = f(y)
    ans = _parallelize(f)(y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testAdd3(self):
    x = np.arange(10)
    y = 2 * np.arange(10)
    def f(x, y):
      return x + y
    expected = f(x, y)
    ans = _parallelize(f)(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @unittest.skip("Missing cases in gather papply rule")
  def testOuter(self):
    x = np.arange(10)
    y = 2 * np.arange(10)
    def f(x): return x[:, None] * y
    expected = f(x)
    ans = _parallelize(f)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testOuter2(self):
    x = np.arange(10)
    y = 2 * np.arange(10)
    def f(y): return x[:, None] * y
    expected = f(y)
    ans = _parallelize(f)(y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @unittest.skip("Missing cases in gather papply rule")
  def testOuter3(self):
    x = np.arange(10)
    y = 2 * np.arange(10)
    def f(x, y): return x[:, None] * y
    expected = f(x, y)
    ans = _parallelize(f)(x, y)
    self.assertAllClose(ans, expected, check_dtypes=False)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "testTranspose_shape={}_perm={}"
       .format(shape, perm),
       "shape": shape, "perm": perm}
      for shape in [
          (2, 2),
          (3, 3),
          (2, 2, 2),
          (2, 3, 4),
          (2, 3, 2)
      ]
      for perm in itertools.permutations(list(range(len(shape))))
  ))
  def testTranspose(self, shape, perm):

    def fun(x):
      return lax.transpose(x, perm)

    x = np.arange(prod(shape)).reshape(shape)
    expected = fun(x)
    ans = _parallelize(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposeAndAddRank2(self):

    def fun(x):
      return x + x.T

    x = np.reshape(np.arange(4., dtype=np.float32), (2, 2))
    expected = fun(x)
    ans = _parallelize(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testTransposeAndAddRank3(self):

    def fun(x):
      return x + x.T

    x = np.reshape(np.arange(8., dtype=np.float32), (2, 2, 2))
    expected = fun(x)
    ans = _parallelize(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testDot(self):
    raise SkipTest("known failure")  # TODO(frostig)
    x = np.reshape(np.arange(4., dtype=np.float32), (2, 2))

    def fun(x, y):
      return lax.dot(x, y)

    expected = fun(x, x)
    pfun, axis_name = _papply(fun)
    ans = soft_pmap(pfun, axis_name)(x, x)
    ans = self.dedup(ans, expected.ndim)
    self.assertAllClose(ans, expected, check_dtypes=False)

  # Test lax.dot_general on two rank-3 arguments, generating a test method call
  # for every matching of dimensions, and each matched pair of dimensions being
  # {batch, contracting, neither}. In combination with that, split the first
  # dimension of the LHS, that of the RHS, and that of both.
  @parameterized.named_parameters(
      {"testcase_name": "_dimMatch={}_matchTypes={}_split={}".format(
          matching, coloring, split),
       "matching": matching, "coloring": coloring, "split": split}
      for matching in itertools.permutations(range(3))
      for coloring in itertools.product(range(3), range(3), range(3))
      for split in range(3))
  def testDotGeneral(self, matching, coloring, split):
    BATCH, CONTRACT, _ = range(3)
    SPLIT_LHS, SPLIT_RHS, SPLIT_BOTH = range(3)

    x = np.reshape(np.arange(8.), (2, 2, 2))
    y = np.reshape(np.arange(8.), (2, 2, 2)) + 4.

    cdims = [(i, matching[i]) for i in range(3) if coloring[i] == CONTRACT]
    bdims = [(i, matching[i]) for i in range(3) if coloring[i] == BATCH]
    dimension_numbers = [
        list(zip(*cdims)) or [(), ()],
        list(zip(*bdims)) or [(), ()]
    ]

    def f(x, y):
      return lax.dot_general(x, y, dimension_numbers)

    if split == SPLIT_LHS:
      fun = lambda x: f(x, y)
    elif split == SPLIT_RHS:
      fun = lambda y: f(x, y)
    else:
      fun = f

    try:
      if split != SPLIT_BOTH:
        expected = fun(x)
        pfun, axis_name = _papply(fun)
        ans = soft_pmap(pfun, axis_name)(x)
      else:
        expected = fun(x, y)
        pfun, axis_name = _papply(fun)
        ans = soft_pmap(pfun, axis_name)(x, y)
    except (NotImplementedError, TypeError) as e:
      raise SkipTest(str(e)) from e

    ans = self.dedup(ans, expected.ndim)
    self.assertAllClose(ans, expected, check_dtypes=False)

  def testCall(self):
    @jit
    def fun(x):
      return x

    x = np.reshape(np.arange(8., dtype=np.float32), (2, 2, 2))
    expected = fun(x)
    ans = _parallelize(fun)(x)
    self.assertAllClose(ans, expected, check_dtypes=False)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
