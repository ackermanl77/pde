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

import numpy as onp

from absl.testing import absltest
from absl.testing import parameterized
import scipy.ndimage as osp_ndimage

from jax import grad
from jax import test_util as jtu
from jax import dtypes
from jax.scipy import ndimage as lsp_ndimage

from jax.config import config
config.parse_flags_with_absl()


float_dtypes = jtu.dtypes.floating
int_dtypes = jtu.dtypes.integer


def _fixed_ref_map_coordinates(input, coordinates, order, mode, cval=0.0):
  # SciPy's implementation of map_coordinates handles boundaries incorrectly,
  # unless mode='reflect'. For order=1, this only affects interpolation outside
  # the bounds of the original array.
  # https://github.com/scipy/scipy/issues/2640
  assert order <= 1
  padding = [(max(-onp.floor(c.min()).astype(int) + 1, 0),
              max(onp.ceil(c.max()).astype(int) + 1 - size, 0))
             for c, size in zip(coordinates, input.shape)]
  shifted_coords = [c + p[0] for p, c in zip(padding, coordinates)]
  pad_mode = {
      'nearest': 'edge', 'mirror': 'reflect', 'reflect': 'symmetric'
  }.get(mode, mode)
  if mode == 'constant':
    padded = onp.pad(input, padding, mode=pad_mode, constant_values=cval)
  else:
    padded = onp.pad(input, padding, mode=pad_mode)
  result = osp_ndimage.map_coordinates(
      padded, shifted_coords, order=order, mode=mode, cval=cval)
  return result


class NdimageTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_coordinates={}_order={}_mode={}_cval={}_impl={}_round={}".format(
          jtu.format_shape_dtype_string(shape, dtype),
          jtu.format_shape_dtype_string(coords_shape, coords_dtype),
          order, mode, cval, impl, round_),
       "rng_factory": rng_factory, "shape": shape,
       "coords_shape": coords_shape, "dtype": dtype,
       "coords_dtype": coords_dtype, "order": order, "mode": mode,
       "cval": cval, "impl": impl, "round_": round_}
      for shape in [(5,), (3, 4), (3, 4, 5)]
      for coords_shape in [(7,), (2, 3, 4)]
      for dtype in float_dtypes + int_dtypes
      for coords_dtype in float_dtypes
      for order in [0, 1]
      for mode in ['wrap', 'constant', 'nearest']
      for cval in ([0, -1] if mode == 'constant' else [0])
      for impl, rng_factory in [
          ("original", partial(jtu.rand_uniform, low=0, high=1)),
          ("fixed", partial(jtu.rand_uniform, low=-0.75, high=1.75)),
      ]
      for round_ in [True, False]))
  def testMapCoordinates(self, shape, dtype, coords_shape, coords_dtype, order,
                         mode, cval, impl, round_, rng_factory):

    def args_maker():
      x = onp.arange(onp.prod(shape), dtype=dtype).reshape(shape)
      coords = [(size - 1) * rng(coords_shape, coords_dtype) for size in shape]
      if round_:
        coords = [c.round().astype(int) for c in coords]
      return x, coords

    rng = rng_factory(self.rng())
    lsp_op = lambda x, c: lsp_ndimage.map_coordinates(
        x, c, order=order, mode=mode, cval=cval)
    impl_fun = (osp_ndimage.map_coordinates if impl == "original"
                else _fixed_ref_map_coordinates)
    osp_op = lambda x, c: impl_fun(x, c, order=order, mode=mode, cval=cval)
    if dtype in float_dtypes:
      epsilon = max([dtypes.finfo(dtypes.canonicalize_dtype(d)).eps
                     for d in [dtype, coords_dtype]])
      self._CheckAgainstNumpy(lsp_op, osp_op, args_maker, tol=100*epsilon)
    else:
      self._CheckAgainstNumpy(lsp_op, osp_op, args_maker, tol=0)

  def testMapCoordinatesErrors(self):
    x = onp.arange(5.0)
    c = [onp.linspace(0, 5, num=3)]
    with self.assertRaisesRegex(NotImplementedError, 'requires order<=1'):
      lsp_ndimage.map_coordinates(x, c, order=2)
    with self.assertRaisesRegex(
        NotImplementedError, 'does not yet support mode'):
      lsp_ndimage.map_coordinates(x, c, order=1, mode='reflect')
    with self.assertRaisesRegex(ValueError, 'sequence of length'):
      lsp_ndimage.map_coordinates(x, [c, c], order=1)

  def testMapCoordinateDocstring(self):
    self.assertIn("Only linear interpolation",
                  lsp_ndimage.map_coordinates.__doc__)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}_order={}".format(onp.dtype(dtype), order),
       "dtype": dtype, "order": order}
      for dtype in float_dtypes + int_dtypes
      for order in [0, 1]))
  def testMapCoordinatesRoundHalf(self, dtype, order):
    x = onp.arange(-3, 3, dtype=dtype)
    c = onp.array([[.5, 1.5, 2.5, 3.5]])
    def args_maker():
      return x, c

    lsp_op = lambda x, c: lsp_ndimage.map_coordinates(x, c, order=order)
    osp_op = lambda x, c: osp_ndimage.map_coordinates(x, c, order=order)
    self._CheckAgainstNumpy(lsp_op, osp_op, args_maker)

  def testContinuousGradients(self):
    # regression test for https://github.com/google/jax/issues/3024

    def loss(delta):
      x = onp.arange(100.0)
      border = 10
      indices = onp.arange(x.size) + delta
      # linear interpolation of the linear function y=x should be exact
      shifted = lsp_ndimage.map_coordinates(x, [indices], order=1)
      return ((x - shifted) ** 2)[border:-border].mean()

    # analytical gradient of (x - (x - delta)) ** 2 is 2 * delta
    self.assertAllClose(grad(loss)(0.5), 1.0, check_dtypes=False)
    self.assertAllClose(grad(loss)(1.0), 2.0, check_dtypes=False)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
