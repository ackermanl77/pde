# Copyright 2019 Google LLC
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

from functools import partial
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

from jax import numpy as jnp
from jax import test_util as jtu, jit

from jax.config import config
config.parse_flags_with_absl()


all_dtypes = jtu.dtypes.floating + jtu.dtypes.integer + jtu.dtypes.complex


# TODO: these tests fail without fixed PRNG seeds.


class TestPolynomial(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_dtype={}_leading={}_trailing={}".format(
       jtu.format_shape_dtype_string((length+leading+trailing,), dtype),
       leading, trailing),
     "dtype": dtype, "rng_factory": rng_factory, "length": length,
     "leading": leading, "trailing": trailing}
    for dtype in all_dtypes
    for rng_factory in [jtu.rand_default]
    for length in [0, 3, 9, 10, 17]
    for leading in [0, 1, 2, 3, 5, 7, 10]
    for trailing in [0, 1, 2, 3, 5, 7, 10]))
  def testRoots(self, dtype, rng_factory, length, leading, trailing):
    rng = rng_factory(np.random.RandomState(0))

    def args_maker():
      p = rng((length,), dtype)
      return jnp.concatenate(
        [jnp.zeros(leading, p.dtype), p, jnp.zeros(trailing, p.dtype)]),

    # order may differ (jnp.sort doesn't deal with complex numbers)
    np_fn = lambda arg: np.sort(jnp.roots(arg))
    np_fn = lambda arg: np.sort(np.roots(arg))
    self._CheckAgainstNumpy(np_fn, np_fn, args_maker, check_dtypes=False,
                            tol=3e-6)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_dtype={}_trailing={}".format(
        jtu.format_shape_dtype_string((length+trailing,), dtype), trailing),
     "dtype": dtype, "rng_factory": rng_factory, "length": length,
     "trailing": trailing}
    for dtype in all_dtypes
    for rng_factory in [jtu.rand_default]
    for length in [0, 1, 3, 10]
    for trailing in [0, 1, 3, 7]))
  def testRootsNostrip(self, length, dtype, rng_factory, trailing):
    rng = rng_factory(np.random.RandomState(0))

    def args_maker():
      p = rng((length,), dtype)
      if length != 0:
        return jnp.concatenate([p, jnp.zeros(trailing, p.dtype)]),
      else:
        # adding trailing would make input invalid (start with zeros)
        return p,

    # order may differ (jnp.sort doesn't deal with complex numbers)
    np_fn = lambda arg: np.sort(jnp.roots(arg, strip_zeros=False))
    np_fn = lambda arg: np.sort(np.roots(arg))
    self._CheckAgainstNumpy(np_fn, np_fn, args_maker,
                            check_dtypes=False, tol=1e-6)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_dtype={}_trailing={}".format(
      jtu.format_shape_dtype_string((length + trailing,), dtype), trailing),
      "dtype": dtype, "rng_factory": rng_factory, "length": length,
      "trailing": trailing}
    for dtype in all_dtypes
    for rng_factory in [jtu.rand_default]
    for length in [0, 1, 3, 10]
    for trailing in [0, 1, 3, 7]))
  # TODO: enable when there is an eigendecomposition implementation
  # for GPU/TPU.
  @jtu.skip_on_devices("gpu", "tpu")
  def testRootsJit(self, length, dtype, rng_factory, trailing):
    rng = rng_factory(np.random.RandomState(0))

    def args_maker():
      p = rng((length,), dtype)
      if length != 0:
        return jnp.concatenate([p, jnp.zeros(trailing, p.dtype)]),
      else:
        # adding trailing would make input invalid (start with zeros)
        return p,

    # order may differ (jnp.sort doesn't deal with complex numbers)
    roots_compiled = jit(partial(jnp.roots, strip_zeros=False))
    np_fn = lambda arg: np.sort(roots_compiled(arg))
    np_fn = lambda arg: np.sort(np.roots(arg))
    # Using strip_zeros=False makes the algorithm less efficient
    # and leads to slightly different values compared ot numpy
    self._CheckAgainstNumpy(np_fn, np_fn, args_maker,
                            check_dtypes=False, tol=1e-6)

  @parameterized.named_parameters(jtu.cases_from_list(
    {"testcase_name": "_dtype={}_zeros={}_nonzeros={}".format(
        jtu.format_shape_dtype_string((zeros+nonzeros,), dtype),
        zeros, nonzeros),
     "zeros": zeros, "nonzeros": nonzeros, "dtype": dtype,
     "rng_factory": rng_factory}
    for dtype in all_dtypes
    for rng_factory in [jtu.rand_default]
    for zeros in [1, 2, 5]
    for nonzeros in [0, 3]))
  @jtu.skip_on_devices("gpu")
  def testRootsInvalid(self, zeros, nonzeros, dtype, rng_factory):
    rng = rng_factory(np.random.RandomState(0))

    # The polynomial coefficients here start with zero and would have to
    # be stripped before computing eigenvalues of the companion matrix.
    # Setting strip_zeros=False skips this check,
    # allowing jit transformation but yielding nan's for these inputs.
    p = jnp.concatenate([jnp.zeros(zeros, dtype), rng((nonzeros,), dtype)])

    if p.size == 1:
      # polynomial = const has no roots
      self.assertTrue(jnp.roots(p, strip_zeros=False).size == 0)
    else:
      self.assertTrue(jnp.any(jnp.isnan(jnp.roots(p, strip_zeros=False))))


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
