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

# flake8: noqa: F401
from . import fft
from . import linalg

from ..interpreters.xla import DeviceArray

from .lax_numpy import (
    ComplexWarning, NINF, NZERO, PZERO, abs, absolute, add, all, allclose,
    alltrue, amax, amin, angle, any, append, arange, arccos, arccosh, arcsin,
    arcsinh, arctan, arctan2, arctanh, argmax, argmin, argsort, argwhere, around,
    array, array_equal, array_repr, array_str, asarray, atleast_1d, atleast_2d,
    atleast_3d, average, bartlett, bfloat16, bincount, bitwise_and, bitwise_not,
    bitwise_or, bitwise_xor, blackman, block, bool_, broadcast_arrays,
    broadcast_to, can_cast, cbrt, cdouble, ceil, character, clip, column_stack,
    complex128, complex64, complex_, complexfloating, compress, concatenate,
    conj, conjugate, convolve, copysign, corrcoef, correlate, cos, cosh,
    count_nonzero, cov, cross, csingle, cumprod, cumproduct, cumsum, deg2rad,
    degrees, diag, diagflat, diag_indices, diag_indices_from, diagonal, diff, digitize, divide, divmod, dot,
    double, dsplit, dstack, dtype, e, ediff1d, einsum, einsum_path, empty,
    empty_like, equal, euler_gamma, exp, exp2, expand_dims, expm1, extract, eye,
    fabs, finfo, fix, flatnonzero, flexible, flip, fliplr, flipud, float16, float32,
    float64, float_, float_power, floating, floor, floor_divide, fmax, fmin,
    fmod, frexp, full, full_like, function, gcd, geomspace, gradient, greater,
    greater_equal, hamming, hanning, heaviside, histogram, histogram_bin_edges,
    hsplit, hstack, hypot, identity, iinfo, imag,
    indices, inexact, in1d, inf, inner, int16, int32, int64, int8, int_, integer,
    isclose, iscomplex, iscomplexobj, isfinite, isin, isinf, isnan, isneginf,
    isposinf, isreal, isrealobj, isscalar, issubdtype, issubsctype, iterable,
    ix_, kaiser, kron, lcm, ldexp, left_shift, less, less_equal, linspace,
    load, log, log10, log1p, log2, logaddexp, logaddexp2, logical_and,
    logical_not, logical_or, logical_xor, logspace, mask_indices, matmul, max,
    maximum, mean, median, meshgrid, min, minimum, mod, moveaxis, msort,
    multiply, nan, nan_to_num, nanargmax, nanargmin, nancumprod, nancumsum,
    nanmedian, nanpercentile, nanquantile,
    nanmax, nanmean, nanmin, nanprod, nanstd, nansum, nanvar, ndarray, ndim,
    negative, newaxis, nextafter, nonzero, not_equal, number, numpy_version,
    object_, ones, ones_like, operator_name, outer, packbits, pad, percentile,
    pi, polyadd, polyder, polymul, polysub, polyval, positive, power, prod, product, promote_types, ptp, quantile,
    rad2deg, radians, ravel, real, reciprocal, remainder, repeat, reshape,
    result_type, right_shift, rint, roll, rollaxis, rot90, round, row_stack,
    save, savez, searchsorted, select, set_printoptions, shape, sign, signbit,
    signedinteger, sin, sinc, single, sinh, size, sometrue, sort, split, sqrt,
    square, squeeze, stack, std, subtract, sum, swapaxes, take, take_along_axis,
    tan, tanh, tensordot, tile, trace, trapz, transpose, tri, tril, tril_indices, tril_indices_from,
    triu, triu_indices, triu_indices_from, true_divide, trunc, uint16, uint32, uint64, uint8, unique,
    unpackbits, unravel_index, unsignedinteger, unwrap, vander, var, vdot, vsplit,
    vstack, where, zeros, zeros_like)

from .polynomial import roots
from .vectorize import vectorize


# Module initialization is encapsulated in a function to avoid accidental
# namespace pollution.
def _init():
  import numpy as np
  from . import lax_numpy
  from .. import util
  # Builds a set of all unimplemented NumPy functions.
  for name, func in util.get_module_functions(np).items():
    if name not in globals():
      globals()[name] = lax_numpy._not_implemented(func)

_init()
del _init
