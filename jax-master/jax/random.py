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

"""JAX pseudo-random number generators (PRNGs).

Example usage:

>>> rng = jax.random.PRNGKey(seed)
>>> for i in range(num_steps):
...   rng, rng_input = jax.random.split(rng)
...   params = compiled_update(rng_input, params, next(batches))

Context:

Among other requirements, the JAX PRNG aims to:
(a) ensure reproducibility,
(b) parallelize well, both in terms of vectorization (generating array values)
and multi-replica, multi-core computation. In particular it should not use
sequencing constraints between random function calls.

The approach is based on:
1. "Parallel random numbers: as easy as 1, 2, 3" (Salmon et al. 2011)
2. "Splittable pseudorandom number generators using cryptographic hashing"
(Claessen et al. 2013)

See also https://github.com/google/jax/blob/master/design_notes/prng.md
for the design and its motivation.
"""


from functools import partial
from typing import Optional, Sequence, Union
import warnings

import numpy as np

from . import lax
from . import numpy as jnp
from . import dtypes
from .api import jit, vmap
from .numpy.lax_numpy import _constant_like, asarray
from jax.lib import xla_bridge
from jax.lib import xla_client
from jax.lib import cuda_prng
from jax import core
from jax import abstract_arrays
from jax.numpy.linalg import cholesky
from jax.interpreters import ad
from jax.interpreters import batching
from jax.interpreters import xla
from jax.util import prod


_UINT_DTYPES = {8: jnp.uint8, 16: jnp.uint16, 32: jnp.uint32, 64: jnp.uint64}


def PRNGKey(seed: int) -> jnp.ndarray:
  """Create a pseudo-random number generator (PRNG) key given an integer seed.

  Args:
    seed: a 64- or 32-bit integer used as the value of the key.

  Returns:
    A PRNG key, which is modeled as an array of shape (2,) and dtype uint32. The
    key is constructed from a 64-bit seed by effectively bit-casting to a pair
    of uint32 values (or from a 32-bit seed by first padding out with zeros).
  """
  if np.shape(seed):
    raise TypeError("PRNGKey seed must be a scalar.")
  convert = lambda k: lax.reshape(lax.convert_element_type(k, np.uint32), [1])
  if isinstance(seed, (int, np.ndarray)):
    # Special handling of raw integer values, which may have be 64bit even
    # when jax_enable_x64=False and we don't want to drop the top 32 bits
    k1 = convert(np.bitwise_and(np.right_shift(seed, 32), 0xFFFFFFFF))
  else:
    k1 = convert(lax.shift_right_logical(seed, lax._const(seed, 32)))
  k2 = convert(jnp.bitwise_and(seed, 0xFFFFFFFF))
  return lax.concatenate([k1, k2], 0)

def _is_prng_key(key: jnp.ndarray) -> bool:
  try:
    return key.shape == (2,) and key.dtype == np.uint32
  except AttributeError:
    return False


### utilities


def _make_rotate_left(dtype):
  if not jnp.issubdtype(dtype, np.integer):
    raise TypeError("_rotate_left only accepts integer dtypes.")
  nbits = np.array(jnp.iinfo(dtype).bits, dtype)

  def _rotate_left(x, d):
    if lax.dtype(d) != lax.dtype(x):
      d = lax.convert_element_type(d, x.dtype)
    return lax.shift_left(x, d) | lax.shift_right_logical(x, nbits - d)
  return _rotate_left


def _bit_stats(bits):
  """This is a debugging function to compute the statistics of bit fields."""
  return np.array([list(map(int, np.binary_repr(x, 64))) for x in bits]).mean(0)


### hash function and split

def _threefry2x32_abstract_eval(*args):
  if any(a.dtype != jnp.uint32 for a in args):
    raise TypeError("Arguments to threefry2x32 must have uint32 type, got {}"
                    .format(args))
  if all(isinstance(arg, abstract_arrays.ShapedArray) for arg in args):
    shape = lax._broadcasting_shape_rule(*args)
    aval = abstract_arrays.ShapedArray(shape, jnp.dtype(jnp.uint32))
  else:
    aval = abstract_arrays.UnshapedArray(jnp.dtype(jnp.uint32))
  return (aval,) * 2

rotate_left = _make_rotate_left(np.uint32)

def apply_round(v, rot):
  v = v[:]
  v[0] = v[0] + v[1]
  v[1] = rotate_left(v[1], rot)
  v[1] = v[0] ^ v[1]
  return v

def rotate_list(xs):
  return xs[1:] + xs[:1]

def rolled_loop_step(i, state):
  x, ks, rotations = state
  for r in rotations[0]:
    x = apply_round(x, r)
  new_x = [x[0] + ks[0], x[1] + ks[1] + asarray(i + 1, dtype=np.uint32)]
  return new_x, rotate_list(ks), rotate_list(rotations)

def _threefry2x32_lowering(key1, key2, x1, x2, use_rolled_loops=True):
  """Apply the Threefry 2x32 hash.

  Args:
    keypair: a pair of 32bit unsigned integers used for the key.
    count: an array of dtype uint32 used for the counts.

  Returns:
    An array of dtype uint32 with the same shape as `count`.
  """
  x = [x1, x2]

  rotations = [np.array([13, 15, 26, 6], dtype=np.uint32),
               np.array([17, 29, 16, 24], dtype=np.uint32)]
  ks = [key1, key2, key1 ^ key2 ^ np.uint32(0x1BD11BDA)]

  x[0] = x[0] + ks[0]
  x[1] = x[1] + ks[1]

  if use_rolled_loops:
    x, _, _ = lax.fori_loop(0, 5, rolled_loop_step, (x, rotate_list(ks), rotations))

  else:
    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[1]
    x[1] = x[1] + ks[2] + np.uint32(1)

    for r in rotations[1]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[2]
    x[1] = x[1] + ks[0] + np.uint32(2)

    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[0]
    x[1] = x[1] + ks[1] + np.uint32(3)

    for r in rotations[1]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[1]
    x[1] = x[1] + ks[2] + np.uint32(4)

    for r in rotations[0]:
      x = apply_round(x, r)
    x[0] = x[0] + ks[2]
    x[1] = x[1] + ks[0] + np.uint32(5)

  return tuple(x)


def _threefry2x32_gpu_translation_rule(c, k1, k2, x1, x2):
  shape = lax.broadcast_shapes(
      c.get_shape(k1).dimensions(), c.get_shape(k2).dimensions(),
      c.get_shape(x1).dimensions(), c.get_shape(x2).dimensions())
  rank = len(shape)
  def _broadcast(x):
    ndims = c.get_shape(x).rank()
    return xla_client.ops.BroadcastInDim(x, shape,
                                         tuple(range(rank - ndims, rank)))
  return cuda_prng.threefry2x32(
      c, (_broadcast(k1), _broadcast(k2)), (_broadcast(x1), _broadcast(x2)))

threefry2x32_p = core.Primitive("threefry2x32")
threefry2x32_p.multiple_results = True
threefry2x32_p.def_impl(partial(xla.apply_primitive, threefry2x32_p))
threefry2x32_p.def_abstract_eval(_threefry2x32_abstract_eval)
batching.defbroadcasting(threefry2x32_p)
xla.translations[threefry2x32_p] = xla.lower_fun(
    partial(_threefry2x32_lowering, use_rolled_loops=False),
    multiple_results=True)
xla.backend_specific_translations['cpu'][threefry2x32_p] = xla.lower_fun(
    partial(_threefry2x32_lowering, use_rolled_loops=True),
    multiple_results=True)
if cuda_prng:
  xla.backend_specific_translations['gpu'][threefry2x32_p] = \
      _threefry2x32_gpu_translation_rule

@jit
def threefry_2x32(keypair, count):
  """Apply the Threefry 2x32 hash.

  Args:
    keypair: a pair of 32bit unsigned integers used for the key.
    count: an array of dtype uint32 used for the counts.

  Returns:
    An array of dtype uint32 with the same shape as `count`.
  """
  key1, key2 = keypair
  if not lax.dtype(key1) == lax.dtype(key2) == lax.dtype(count) == np.uint32:
    msg = "threefry_2x32 requires uint32 arguments, got {}"
    raise TypeError(msg.format([lax.dtype(x) for x in [key1, key2, count]]))

  odd_size = count.size % 2
  if odd_size:
    x = list(jnp.split(jnp.concatenate([count.ravel(), np.uint32([0])]), 2))
  else:
    x = list(jnp.split(count.ravel(), 2))

  x = threefry2x32_p.bind(key1, key2, x[0], x[1])
  out = jnp.concatenate(x)
  assert out.dtype == np.uint32
  return lax.reshape(out[:-1] if odd_size else out, count.shape)


def split(key: jnp.ndarray, num: int = 2) -> jnp.ndarray:
  """Splits a PRNG key into `num` new keys by adding a leading axis.

  Args:
    key: a PRNGKey (an array with shape (2,) and dtype uint32).
    num: optional, a positive integer indicating the number of keys to produce
      (default 2).

  Returns:
    An array with shape (num, 2) and dtype uint32 representing `num` new keys.
  """
  return _split(key, num)

@partial(jit, static_argnums=(1,))
def _split(key, num):
  counts = lax.tie_in(key, lax.iota(np.uint32, num * 2))
  return lax.reshape(threefry_2x32(key, counts), (num, 2))


def fold_in(key, data):
  """Folds in data to a PRNG key to form a new PRNG key.

  Args:
    key: a PRNGKey (an array with shape (2,) and dtype uint32).
    data: a 32bit integer representing data to be folded in to the key.

  Returns:
    A new PRNGKey that is a deterministic function of the ijnputs and is
    statistically safe for producing a stream of new pseudo-random values.
  """
  return _fold_in(key, data)

@jit
def _fold_in(key, data):
  key2 = lax.tie_in(key, PRNGKey(data))
  return threefry_2x32(key, key2)


def _random_bits(key, bit_width, shape):
  """Sample uniform random bits of given width and shape using PRNG key."""
  if not _is_prng_key(key):
    raise TypeError("_random_bits got invalid prng key.")
  if bit_width not in (8, 16, 32, 64):
    raise TypeError("requires 8-, 16-, 32- or 64-bit field width.")
  size = np.prod(shape)
  max_count = int(np.ceil(bit_width * size / 32))
  if max_count >= jnp.iinfo(np.uint32).max:
    # TODO(mattjj): just split the key here
    raise TypeError("requesting more random bits than a single call provides.")

  counts = lax.tie_in(key, lax.iota(np.uint32, max_count))
  bits = threefry_2x32(key, counts)
  dtype = _UINT_DTYPES[bit_width]
  if bit_width == 64:
    bits = [lax.convert_element_type(x, dtype) for x in jnp.split(bits, 2)]
    bits = lax.shift_left(bits[0], dtype(32)) | bits[1]
  elif bit_width in [8, 16]:
    # this is essentially bits.view(dtype)[:size]
    bits = lax.bitwise_and(
      np.uint32(np.iinfo(dtype).max),
      lax.shift_right_logical(
        lax.broadcast(bits, (1,)),
        lax.mul(
          np.uint32(bit_width),
          lax.broadcasted_iota(np.uint32, (32 // bit_width, 1), 0)
        )
      )
    )
    bits = lax.reshape(bits, (np.uint32(max_count * 32 // bit_width),), (1, 0))
    bits = lax.convert_element_type(bits, dtype)[:size]
  return lax.reshape(bits, shape)


### random samplers


def _check_shape(name, shape, *param_shapes):
  shape = abstract_arrays.canonicalize_shape(shape)

  if param_shapes:
    shape_ = lax.broadcast_shapes(shape, *param_shapes)
    if shape != shape_:
      msg = ("{} parameter shapes must be broadcast-compatible with shape "
             "argument, and the result of broadcasting the shapes must equal "
             "the shape argument, but got result {} for shape argument {}.")
      raise ValueError(msg.format(name, shape_, shape))


def uniform(key: jnp.ndarray,
            shape: Sequence[int] = (),
            dtype: np.dtype = np.float64,
            minval: Union[float, jnp.ndarray] = 0.,
            maxval: Union[float, jnp.ndarray] = 1.) -> jnp.ndarray:
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    minval: optional, a minimum (inclusive) value for the range (default 0).
    maxval: optional, a maximum (exclusive) value for the range (default 1).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `uniform` must be a float dtype, "
                     f"got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _uniform(key, shape, dtype, minval, maxval)

@partial(jit, static_argnums=(1, 2))
def _uniform(key, shape, dtype, minval, maxval):
  _check_shape("uniform", shape)
  if not jnp.issubdtype(dtype, np.floating):
    raise TypeError("uniform only accepts floating point dtypes.")

  minval = lax.convert_element_type(minval, dtype)
  maxval = lax.convert_element_type(maxval, dtype)
  finfo = jnp.finfo(dtype)
  nbits, nmant = finfo.bits, finfo.nmant

  if nbits not in (16, 32, 64):
    raise TypeError("uniform only accepts 32- or 64-bit dtypes.")

  bits = _random_bits(key, nbits, shape)

  # The strategy here is to randomize only the mantissa bits with an exponent of
  # 1 (after applying the bias), then shift and scale to the desired range. The
  # bit-level transformation we use relies on Numpy and XLA having bit-for-bit
  # equivalent float representations, which might not be true on all platforms.
  float_bits = lax.bitwise_or(
      lax.shift_right_logical(bits, np.array(nbits - nmant, lax.dtype(bits))),
      np.array(1., dtype).view(_UINT_DTYPES[nbits]))
  floats = lax.bitcast_convert_type(float_bits, dtype) - np.array(1., dtype)
  return lax.max(
      minval,
      lax.reshape(floats * (maxval - minval) + minval, shape))


def randint(key: jnp.ndarray,
            shape: Sequence[int],
            minval: Union[int, jnp.ndarray],
            maxval: Union[int, jnp.ndarray],
            dtype: np.dtype = np.int64):
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: a tuple of nonnegative integers representing the shape.
    minval: int or array of ints broadcast-compatible with ``shape``, a minimum
      (inclusive) value for the range.
    maxval: int or array of ints broadcast-compatible with ``shape``, a maximum
      (exclusive) value for the range.
    dtype: optional, an int dtype for the returned values (default int64 if
      jax_enable_x64 is true, otherwise int32).

  Returns:
    A random array with the specified shape and dtype.
  """
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _randint(key, shape, minval, maxval, dtype)

@partial(jit, static_argnums=(1, 4))
def _randint(key, shape, minval, maxval, dtype):
  _check_shape("randint", shape, np.shape(minval), np.shape(maxval))
  if not jnp.issubdtype(dtype, np.integer):
    raise TypeError("randint only accepts integer dtypes.")

  minval = lax.convert_element_type(minval, dtype)
  maxval = lax.convert_element_type(maxval, dtype)
  nbits = jnp.iinfo(dtype).bits

  if nbits not in (8, 16, 32, 64):
    raise TypeError("randint only accepts 8-, 16-, 32-, or 64-bit dtypes.")

  # if we don't have minval < maxval, just always return minval
  # https://github.com/google/jax/issues/222
  maxval = lax.max(lax.add(minval, np.array(1, dtype)), maxval)

  # This algorithm is biased whenever (maxval - minval) is not a power of 2.
  # We generate double the number of random bits required by the dtype so as to
  # reduce that bias.
  k1, k2 = split(key)
  rbits = lambda key: _random_bits(key, nbits, shape)
  higher_bits, lower_bits = rbits(k1), rbits(k2)

  unsigned_dtype = _UINT_DTYPES[nbits]
  span = lax.convert_element_type(maxval - minval, unsigned_dtype)

  # To compute a remainder operation on an integer that might have twice as many
  # bits as we can represent in the native unsigned dtype, we compute a
  # multiplier equal to 2**nbits % span. To avoid overflow, we use the identity:
  #  (a * b) % N = [(a % N) * (b % N)] % N
  multiplier = lax.rem(lax._const(span, 2 ** (nbits // 2)), span)
  multiplier = lax.rem(lax.mul(multiplier, multiplier), span)

  random_offset = lax.add(lax.mul(lax.rem(higher_bits, span), multiplier),
                          lax.rem(lower_bits, span))
  random_offset = lax.rem(random_offset, span)
  return lax.add(minval, lax.convert_element_type(random_offset, dtype))


def shuffle(key: jnp.ndarray, x: jnp.ndarray, axis: int = 0) -> jnp.ndarray:
  """Shuffle the elements of an array uniformly at random along an axis.

  Args:
    key: a PRNGKey used as the random key.
    x: the array to be shuffled.
    axis: optional, an int axis along which to shuffle (default 0).

  Returns:
    A shuffled version of x.
  """
  msg = ("jax.random.shuffle is deprecated and will be removed in a future release. "
         "Use jax.random.permutation")
  warnings.warn(msg, FutureWarning)
  return _shuffle(key, x, axis)


def permutation(key, x):
  """
  Permute elements of an array along its first axis or return a permuted range.

  If `x` is a multi-dimensional array, it is only shuffled along its
  first index.

  Args:n
    key: a PRNGKey used as the random key.
    x: the array or integer range to be shuffled.

  Returns:
    A shuffled version of x or array range
  """
  if not np.ndim(x):
    # scalar case, must be a concrete integer
    if not np.issubdtype(lax.dtype(x), np.integer):
      raise TypeError("x must be an integer or at least 1-dimensional")
    x = int(x)
    return _shuffle(key, jnp.arange(x), 0)
  elif np.ndim(x) == 1:
    return _shuffle(key, x, 0)
  else:
    ind = _shuffle(key, jnp.arange(x.shape[0]), 0)
    return x[ind]


@partial(jit, static_argnums=(2,))
def _shuffle(key, x, axis):
  # On parallel architectures, Fisher-Yates is more expensive than doing
  # multiple sorts. This algorithm is based on one developed and analyzed by
  # tjablin@. We sort according to randomly-generated 32bit keys, but those keys
  # may have collisions. If we repeat the process, using fresh 32bit keys for
  # each sort, then whenever all pairs of elements have been assigned distinct
  # keys at some iteration (or equivalently when the strings formed by
  # concatenating the successive keys for each element are all distinct) then we
  # are guaranteed to have a perfect sample (assuming that either the sort is
  # stable or that any bias is not value-dependent). Since checking uniqueness
  # at runtime may be expensive, we use a heuristic static stop criterion
  # developed by tjablin@. See tensorflow/compiler/tf2xla/random_ops.cc for more
  # info, and for the original implementation of this algorithm. See also
  # Section 2 of http://people.csail.mit.edu/costis/6896sp11/lec5s.pdf for
  # another analysis (where the keys are generated one bit at a time).
  exponent = 3  # see tjablin@'s analysis for explanation of this parameter
  uint32max = jnp.iinfo(np.uint32).max
  num_rounds = int(np.ceil(exponent * np.log(x.size) / np.log(uint32max)))

  for _ in range(num_rounds):
    key, subkey = split(key)
    sort_keys = _random_bits(subkey, 32, x.shape)
    _, x = lax.sort_key_val(sort_keys, x, axis)

  return x


def choice(key, a, shape=(), replace=True, p=None):
  """Generates a random sample from a given 1-D array.

  Args:
    key: a PRNGKey used as the random key.
    a : 1D array or int. If an ndarray, a random sample is generated from
      its elements. If an int, the random sample is generated as if a were
      arange(a).
    shape : tuple of ints, optional. Output shape.  If the given shape is,
      e.g., ``(m, n)``, then ``m * n`` samples are drawn.  Default is (),
      in which case a single value is returned.
    replace : boolean.  Whether the sample is with or without replacement.
      default is True.
    p : 1-D array-like, The probabilities associated with each entry in a.
      If not given the sample assumes a uniform distribution over all
      entries in a.

  Returns:
    An array of shape `shape` containing samples from `a`.
  """
  a = jnp.asarray(a)
  if a.ndim not in [0, 1]:
    raise ValueError("a must be an integer or 1-dimensional")
  n_inputs = int(a) if a.ndim == 0 else len(a)
  n_draws = np.prod(shape).astype(int)
  if n_draws == 0:
    return jnp.zeros(shape, dtype=a.dtype)
  if n_inputs <= 0:
    raise ValueError("a must be greater than 0 unless no samples are taken")
  if not replace and n_draws > n_inputs:
    raise ValueError("Cannot take a larger sample than population when 'replace=False'")

  if p is None:
    if replace:
      ind = randint(key, shape, 0, n_inputs)
      result = ind if a.ndim == 0 else a[ind]
    else:
      result = permutation(key, a)[:n_draws]
  else:
    p = jnp.asarray(p)
    if p.shape != (n_inputs,):
      raise ValueError("p must be None or match the shape of a")
    if replace:
      p_cuml = jnp.cumsum(p)
      r = p_cuml[-1] * (1 - uniform(key, shape))
      ind = jnp.searchsorted(p_cuml, r)
      result = ind if a.ndim == 0 else a[ind]
    else:
      # Gumbel top-k trick: https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
      g = -gumbel(key, (n_inputs,)) - jnp.log(p)
      ind = jnp.argsort(g)[:n_draws]
      result = ind if a.ndim == 0 else a[ind]
  return result.reshape(shape)


def normal(key: jnp.ndarray,
           shape: Sequence[int] = (),
           dtype: np.dtype = np.float64) -> jnp.ndarray:
  """Sample standard normal random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `normal` must be a float dtype, "
                     f"got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _normal(key, shape, dtype)

@partial(jit, static_argnums=(1, 2))
def _normal(key, shape, dtype):
  _check_shape("normal", shape)
  lo = np.nextafter(np.array(-1., dtype), 0., dtype=dtype)
  hi = np.array(1., dtype)
  u = uniform(key, shape, dtype, lo, hi)
  return np.array(np.sqrt(2), dtype) * lax.erf_inv(u)


def multivariate_normal(key: jnp.ndarray,
                        mean: jnp.ndarray,
                        cov: jnp.ndarray,
                        shape: Optional[Sequence[int]] = None,
                        dtype: np.dtype = np.float64) -> jnp.ndarray:
  """Sample multivariate normal random values with given mean and covariance.

  Args:
    key: a PRNGKey used as the random key.
    mean: a mean vector of shape ``(..., n)``.
    cov: a positive definite covariance matrix of shape ``(..., n, n)``. The
      batch shape ``...`` must be broadcast-compatible with that of ``mean``.
    shape: optional, a tuple of nonnegative integers specifying the result
      batch shape; that is, the prefix of the result shape excluding the last
      axis. Must be broadcast-compatible with ``mean.shape[:-1]`` and
      ``cov.shape[:-2]``. The default (None) produces a result batch shape by
      broadcasting together the batch shapes of ``mean`` and ``cov``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and shape given by
    ``shape + mean.shape[-1:]`` if ``shape`` is not None, or else
    ``broadcast_shapes(mean.shape[:-1], cov.shape[:-2]) + mean.shape[-1:]``.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `multivariate_normal` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  if shape is not None:
    shape = abstract_arrays.canonicalize_shape(shape)
  return _multivariate_normal(key, mean, cov, shape, dtype)

@partial(jit, static_argnums=(3, 4))
def _multivariate_normal(key, mean, cov, shape, dtype):
  if not np.ndim(mean) >= 1:
    msg = "multivariate_normal requires mean.ndim >= 1, got mean.ndim == {}"
    raise ValueError(msg.format(np.ndim(mean)))
  if not np.ndim(cov) >= 2:
    msg = "multivariate_normal requires cov.ndim >= 2, got cov.ndim == {}"
    raise ValueError(msg.format(np.ndim(cov)))
  n = mean.shape[-1]
  if np.shape(cov)[-2:] != (n, n):
    msg = ("multivariate_normal requires cov.shape == (..., n, n) for n={n}, "
           "but got cov.shape == {shape}.")
    raise ValueError(msg.format(n=n, shape=np.shape(cov)))

  if shape is None:
    shape = lax.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
  else:
    _check_shape("normal", shape, mean.shape[:-1], mean.shape[:-2])

  chol_factor = cholesky(cov)
  normal_samples = normal(key, shape + mean.shape[-1:], dtype)
  return mean + jnp.tensordot(normal_samples, chol_factor, [-1, 1])


def truncated_normal(key: jnp.ndarray,
                    lower: Union[float, jnp.ndarray],
                    upper: Union[float, jnp.ndarray],
                    shape: Optional[Sequence[int]] = None,
                    dtype: np.dtype = np.float64) -> jnp.ndarray:
  """Sample truncated standard normal random values with given shape and dtype.

  Args:
    key: a PRNGKey used as the random key.
    lower: a float or array of floats representing the lower bound for
      truncation. Must be broadcast-compatible with ``upper``.
    upper: a float or array of floats representing the  upper bound for
      truncation. Must be broadcast-compatible with ``lower``.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``lower`` and ``upper``. The
      default (None) produces a result shape by broadcasting ``lower`` and
      ``upper``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``lower`` and ``upper``.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `truncated_normal` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  if shape is not None:
    shape = abstract_arrays.canonicalize_shape(shape)
  return _truncated_normal(key, lower, upper, shape, dtype)

@partial(jit, static_argnums=(3, 4))
def _truncated_normal(key, lower, upper, shape, dtype):
  if shape is None:
    shape = lax.broadcast_shapes(np.shape(lower), np.shape(upper))
  else:
    _check_shape("truncated_normal", shape, np.shape(lower), np.shape(upper))

  sqrt2 = np.array(np.sqrt(2), dtype)
  a = lax.erf(lax.convert_element_type(lower, dtype) / sqrt2)
  b = lax.erf(lax.convert_element_type(upper, dtype) / sqrt2)
  if not jnp.issubdtype(dtype, np.floating):
    raise TypeError("truncated_normal only accepts floating point dtypes.")
  u = uniform(key, shape, dtype, minval=jnp.finfo(dtype).tiny)
  return sqrt2 * lax.erf_inv(a + u * (b - a))


def bernoulli(key: jnp.ndarray,
              p: jnp.ndarray = np.float32(0.5),
              shape: Optional[Sequence[int]] = None) -> jnp.ndarray:
  """Sample Bernoulli random values with given shape and mean.

  Args:
    key: a PRNGKey used as the random key.
    p: optional, a float or array of floats for the mean of the random
      variables. Must be broadcast-compatible with ``shape``. Default 0.5.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Must be broadcast-compatible with ``p.shape``. The default (None)
      produces a result shape equal to ``p.shape``.

  Returns:
    A random array with boolean dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``p.shape``.
  """
  dtype = dtypes.canonicalize_dtype(lax.dtype(p))
  if shape is not None:
    shape = abstract_arrays.canonicalize_shape(shape)
  if not jnp.issubdtype(dtype, np.floating):
    msg = "bernoulli probability `p` must have a floating dtype, got {}."
    raise TypeError(msg.format(dtype))
  p = lax.convert_element_type(p, dtype)
  return _bernoulli(key, p, shape)

@partial(jit, static_argnums=(2,))
def _bernoulli(key, p, shape):
  if shape is None:
    shape = np.shape(p)
  else:
    _check_shape("bernoulli", shape, np.shape(p))

  return uniform(key, shape, lax.dtype(p)) < p


def beta(key: jnp.ndarray,
         a: Union[float, jnp.ndarray],
         b: Union[float, jnp.ndarray],
         shape: Optional[Sequence[int]] = None,
         dtype: np.dtype = np.float64) -> jnp.ndarray:
  """Sample Beta random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the first parameter "alpha".
    b: a float or array of floats broadcast-compatible with ``shape``
      representing the second parameter "beta".
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``a`` and ``b``. The default
      (None) produces a result shape by broadcasting ``a`` and ``b``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``a`` and ``b``.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `beta` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  if shape is not None:
    shape = abstract_arrays.canonicalize_shape(shape)
  return _beta(key, a, b, shape, dtype)

def _beta(key, a, b, shape, dtype):
  if shape is None:
    shape = lax.broadcast_shapes(np.shape(a), np.shape(b))
  else:
    _check_shape("beta", shape, np.shape(a), np.shape(b))

  a = lax.convert_element_type(a, dtype)
  b = lax.convert_element_type(b, dtype)
  key_a, key_b = split(key)
  a = jnp.broadcast_to(a, shape)
  b = jnp.broadcast_to(b, shape)
  gamma_a = gamma(key_a, a, shape, dtype)
  gamma_b = gamma(key_b, b, shape, dtype)
  return gamma_a / (gamma_a + gamma_b)


def cauchy(key, shape=(), dtype=np.float64):
  """Sample Cauchy random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `cauchy` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _cauchy(key, shape, dtype)

@partial(jit, static_argnums=(1, 2))
def _cauchy(key, shape, dtype):
  _check_shape("cauchy", shape)
  u = uniform(key, shape, dtype, minval=jnp.finfo(dtype).eps, maxval=1.)
  pi = _constant_like(u, np.pi)
  return lax.tan(lax.mul(pi, lax.sub(u, _constant_like(u, 0.5))))


def dirichlet(key, alpha, shape=None, dtype=np.float64):
  """Sample Dirichlet random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    alpha: an array of shape ``(..., n)`` used as the concentration
      parameter of the random variables.
    shape: optional, a tuple of nonnegative integers specifying the result
      batch shape; that is, the prefix of the result shape excluding the last
      element of value ``n``. Must be broadcast-compatible with
      ``alpha.shape[:-1]``. The default (None) produces a result shape equal to
      ``alpha.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and shape given by
    ``shape + (alpha.shape[-1],)`` if ``shape`` is not None, or else
    ``alpha.shape``.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `dirichlet` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  if shape is not None:
    shape = abstract_arrays.canonicalize_shape(shape)
  return _dirichlet(key, alpha, shape, dtype)

@partial(jit, static_argnums=(2, 3))
def _dirichlet(key, alpha, shape, dtype):
  if not np.ndim(alpha) >= 1:
    msg = "dirichlet requires alpha.ndim >= 1, got alpha.ndim == {}"
    raise ValueError(msg.format(np.ndim(alpha)))

  if shape is None:
    shape = np.shape(alpha)[:-1]
  else:
    _check_shape("dirichlet", shape, np.shape(alpha)[:-1])

  alpha = lax.convert_element_type(alpha, dtype)
  gamma_samples = gamma(key, alpha, shape + np.shape(alpha)[-1:], dtype)
  return gamma_samples / jnp.sum(gamma_samples, axis=-1, keepdims=True)


def exponential(key, shape=(), dtype=np.float64):
  """Sample Exponential random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `exponential` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _exponential(key, shape, dtype)

@partial(jit, static_argnums=(1, 2))
def _exponential(key, shape, dtype):
  _check_shape("exponential", shape)
  u = uniform(key, shape, dtype)
  # taking 1 - u to move the domain of log to (0, 1] instead of [0, 1)
  return lax.neg(lax.log1p(lax.neg(u)))


def _gamma_one(key, alpha):
  # Ref: A simple method for generating gamma variables, George Marsaglia and Wai Wan Tsang
  # The algorithm can also be founded in:
  # https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables
  zero = _constant_like(alpha, 0)
  one = _constant_like(alpha, 1)
  minus_one = _constant_like(alpha, -1)
  one_over_two = _constant_like(alpha, 0.5)
  one_over_three = _constant_like(alpha, 1. / 3.)
  squeeze_const = _constant_like(alpha, 0.0331)
  dtype = lax.dtype(alpha)

  key, subkey = split(key)
  # for alpha < 1, we boost alpha to alpha + 1 and get a sample according to
  # Gamma(alpha) ~ Gamma(alpha+1) * Uniform()^(1 / alpha)
  boost = lax.select(lax.ge(alpha, one),
                     one,
                     lax.pow(uniform(subkey, (), dtype=dtype), lax.div(one, alpha)))
  alpha = lax.select(lax.ge(alpha, one), alpha, lax.add(alpha, one))

  d = lax.sub(alpha, one_over_three)
  c = lax.div(one_over_three, lax.pow(d, one_over_two))

  def _cond_fn(kXVU):
    _, X, V, U = kXVU
    # TODO: use lax.cond when its batching rule is supported
    # The reason is to avoid evaluating second condition which involves log+log
    # if the first condition is satisfied
    cond = lax.bitwise_and(lax.ge(U, lax.sub(one, lax.mul(squeeze_const, lax.mul(X, X)))),
                           lax.ge(lax.log(U), lax.add(lax.mul(X, one_over_two),
                                                      lax.mul(d, lax.add(lax.sub(one, V),
                                                                         lax.log(V))))))
    return cond

  def _body_fn(kXVU):
    def _next_kxv(kxv):
      key = kxv[0]
      key, subkey = split(key)
      x = normal(subkey, (), dtype=dtype)
      v = lax.add(one, lax.mul(x, c))
      return key, x, v

    key = kXVU[0]
    key, x_key, U_key = split(key, 3)
    _, x, v = lax.while_loop(lambda kxv: lax.le(kxv[2], zero), _next_kxv, (x_key, zero, minus_one))
    X = lax.mul(x, x)
    V = lax.mul(lax.mul(v, v), v)
    U = uniform(U_key, (), dtype=dtype)
    return key, X, V, U

  # initial state is chosen such that _cond_fn will return True
  _, _, V, _ = lax.while_loop(_cond_fn, _body_fn, (key, zero, one, _constant_like(alpha, 2)))
  z = lax.mul(lax.mul(d, V), boost)
  return lax.select(lax.eq(z, zero), jnp.finfo(z.dtype).tiny, z)


def _gamma_grad(sample, a):
  samples = jnp.reshape(sample, -1)
  alphas = jnp.reshape(a, -1)
  if xla_bridge.get_backend().platform == 'cpu':
    grads = lax.map(lambda args: lax.random_gamma_grad(*args), (alphas, samples))
  else:
    grads = vmap(lax.random_gamma_grad)(alphas, samples)
  return grads.reshape(np.shape(a))

def _gamma_impl(key, a):
  a_shape = jnp.shape(a)
  # split key to match the shape of a
  key_ndim = jnp.ndim(key) - 1
  key = jnp.reshape(key, (-1, 2))
  key = vmap(split, in_axes=(0, None))(key, prod(a_shape[key_ndim:]))
  keys = jnp.reshape(key, (-1, 2))
  alphas = jnp.reshape(a, -1)
  if xla_bridge.get_backend().platform == 'cpu':
    samples = lax.map(lambda args: _gamma_one(*args), (keys, alphas))
  else:
    samples = vmap(_gamma_one)(keys, alphas)
  return jnp.reshape(samples, a_shape)

def _gamma_batching_rule(batched_args, batch_dims):
    k, a = batched_args
    bk, ba = batch_dims
    size = next(t.shape[i] for t, i in zip(batched_args, batch_dims) if i is not None)
    k = batching.bdim_at_front(k, bk, size)
    a = batching.bdim_at_front(a, ba, size)
    return random_gamma_p.bind(k, a), 0

random_gamma_p = core.Primitive('random_gamma')
random_gamma_p.def_impl(_gamma_impl)
random_gamma_p.def_abstract_eval(lambda key, a: abstract_arrays.raise_to_shaped(a))
ad.defjvp2(random_gamma_p, None, lambda tangent, ans, key, a: tangent * _gamma_grad(ans, a))
xla.translations[random_gamma_p] = xla.lower_fun(_gamma_impl, multiple_results=False)
batching.primitive_batchers[random_gamma_p] = _gamma_batching_rule

def gamma(key, a, shape=None, dtype=np.float64):
  """Sample Gamma random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``a``. The default (None)
      produces a result shape equal to ``a.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``a.shape``.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `gamma` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  if shape is not None:
    shape = abstract_arrays.canonicalize_shape(shape)
  return _gamma(key, a, shape, dtype)

@partial(jit, static_argnums=(2, 3))
def _gamma(key, a, shape, dtype):
  if shape is None:
    shape = np.shape(a)
  else:
    _check_shape("gamma", shape, np.shape(a))

  a = lax.convert_element_type(a, dtype)
  if np.shape(a) != shape:
    a = jnp.broadcast_to(a, shape)
  return random_gamma_p.bind(key, a)


@partial(jit, static_argnums=(2, 3, 4))
def _poisson_knuth(key, lam, shape, dtype, max_iters):
  # Knuth's algorithm for generating Poisson random variates.
  # Reference:
  # https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables

  def body_fn(carry):
    i, k, rng, log_prod = carry
    rng, subkey = split(rng)
    k = lax.select(log_prod > -lam, k + 1, k)
    u = uniform(subkey, shape, np.float32)
    return i + 1, k, rng, log_prod + jnp.log(u)

  def cond_fn(carry):
    i, log_prod = carry[0], carry[3]
    return (log_prod > -lam).any() & (i < max_iters)

  k_init = lax.full_like(lam, 0, dtype, shape)
  log_rate_init = lax.full_like(lam, 0, np.float32, shape)
  k = lax.while_loop(cond_fn, body_fn, (0, k_init, key, log_rate_init))[1]
  return (k - 1).astype(dtype)


@partial(jit, static_argnums=(2, 3, 4))
def _poisson_rejection(key, lam, shape, dtype, max_iters):
  # Transformed rejection due to Hormann.
  # Reference:
  # http://citeseer.ist.psu.edu/viewdoc/citations;jsessionid=1BEB35946CC807879F55D42512E5490C?doi=10.1.1.48.3054.
  log_lam = lax.log(lam)
  b = 0.931 + 2.53 * lax.sqrt(lam)
  a = -0.059 + 0.02483 * b
  inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
  v_r = 0.9277 - 3.6224 / (b - 2)

  def body_fn(carry):
    i, k_out, accepted, key = carry
    key, subkey_0, subkey_1 = split(key, 3)

    u = uniform(subkey_0, shape, lam.dtype) - 0.5
    v = uniform(subkey_1, shape, lam.dtype)
    u_shifted = 0.5 - abs(u)

    k = lax.floor((2 * a / u_shifted + b) * u + lam + 0.43)
    s = lax.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
    t = -lam + k * log_lam - lax.lgamma(k + 1)

    accept1 = (u_shifted >= 0.07) & (v <= v_r)
    reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
    accept2 = s <= t
    accept = accept1 | (~reject & accept2)

    k_out = lax.select(accept, k, k_out)
    accepted |= accept

    return i + 1, k_out, accepted, key

  def cond_fn(carry):
    i, k_out, accepted, key = carry
    return (~accepted).any() & (i < max_iters)

  k_init = lax.full_like(lam, -1, lam.dtype, shape)
  accepted = lax.full_like(lam, False, jnp.bool_, shape)
  k = lax.while_loop(cond_fn, body_fn, (0, k_init, accepted, key))[1]
  return k.astype(dtype)


@partial(jit, static_argnums=(2, 3))
def _poisson(key, lam, shape, dtype):
  # The implementation matches TensorFlow and NumPy:
  # https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/core/kernels/random_poisson_op.cc
  # https://github.com/numpy/numpy/blob/v1.18.3/numpy/random/src/distributions/distributions.c#L574
  # For lambda < 10, we use the Knuth algorithm; otherwise, we use transformed
  # rejection sampling.
  use_knuth = lam < 10
  lam_knuth = lax.select(use_knuth, lam, lax.full_like(lam, 0.0))
  # The acceptance probability for rejection sampling maxes out at 89% as
  # λ -> ∞, so pick some arbitrary large value.
  lam_rejection = lax.select(use_knuth, lax.full_like(lam, 1e5), lam)
  max_iters = jnp.iinfo(dtype).max  # insanely conservative
  return lax.select(
      use_knuth,
      _poisson_knuth(key, lam_knuth, shape, dtype, max_iters),
      _poisson_rejection(key, lam_rejection, shape, dtype, max_iters),
  )


def poisson(key, lam, shape=(), dtype=np.int64):
  """Sample Poisson random values with given shape and integer dtype.

  Args:
    key: a PRNGKey used as the random key.
    lam: rate parameter (mean of the distribution), must be >= 0.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a integer dtype for the returned values (default int64 if
      jax_enable_x64 is true, otherwise int32).

  Returns:
    A random array with the specified shape and dtype.
  """
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  if np.shape(lam) != shape:
    lam = jnp.broadcast_to(lam, shape)
  lam = lax.convert_element_type(lam, np.float32)
  return _poisson(key, lam, shape, dtype)


def gumbel(key, shape=(), dtype=np.float64):
  """Sample Gumbel random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `gumbel` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _gumbel(key, shape, dtype)

@partial(jit, static_argnums=(1, 2))
def _gumbel(key, shape, dtype):
  _check_shape("gumbel", shape)
  return -jnp.log(-jnp.log(
      uniform(key, shape, dtype, minval=jnp.finfo(dtype).eps, maxval=1.)))


def categorical(key, logits, axis=-1, shape=None):
  """Sample random values from categorical distributions.

  Args:
    key: a PRNGKey used as the random key.
    logits: Unnormalized log probabilities of the categorical distribution(s) to sample from,
      so that `softmax(logits, axis)` gives the corresponding probabilities.
    axis: Axis along which logits belong to the same categorical distribution.
    shape: Optional, a tuple of nonnegative integers representing the result shape.
      Must be broadcast-compatible with ``np.delete(logits.shape, axis)``.
      The default (None) produces a result shape equal to ``np.delete(logits.shape, axis)``.

  Returns:
    A random array with int dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``np.delete(logits.shape, axis)``.
  """

  if axis >= 0:
    axis -= len(logits.shape)

  batch_shape = tuple(np.delete(logits.shape, axis))
  if shape is None:
    shape = batch_shape
  else:
    _check_shape("categorical", shape, batch_shape)

  sample_shape = shape[:len(shape)-len(batch_shape)]
  return jnp.argmax(gumbel(key, sample_shape + logits.shape, logits.dtype) + logits, axis=axis)


def laplace(key, shape=(), dtype=np.float64):
  """Sample Laplace random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `laplace` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _laplace(key, shape, dtype)

@partial(jit, static_argnums=(1, 2))
def _laplace(key, shape, dtype):
  _check_shape("laplace", shape)
  u = uniform(
      key, shape, dtype, minval=-1. + jnp.finfo(dtype).epsneg, maxval=1.)
  return lax.mul(lax.sign(u), lax.log1p(lax.neg(lax.abs(u))))


def logistic(key, shape=(), dtype=np.float64):
  """Sample logistic random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified shape and dtype.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `logistic` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _logistic(key, shape, dtype)

@partial(jit, static_argnums=(1, 2))
def _logistic(key, shape, dtype):
  # Mathematically, we can compute the distribution by generating uniformly-distributed
  # numbers x in the open interval (a, b) and computing:
  #   z = log[ (x - a) / (b - x))
  # It's important to avoid x=a or x=b, which lead to infinite values for z.
  # The uniform() function generates pseudorandom floating point numbers x in the
  # semi-closed interval [0, 1), so if used directly  with (a,b)=(0,1), it will
  # lead to infinite output in a small number of cases (as many as 1 in 2^23 for float32).
  #
  # Instead, we let (a, b) = (-ε, 1) where ε is the smallest step between floating point
  # values: then numbers in the interval (-ε, 1) are approximated by standard uniformly
  # drawn numbers in [0, 1).
  _check_shape("logistic", shape)
  x = uniform(key, shape, dtype)
  eps = jnp.finfo(dtype).eps
  return lax.log(lax.div(lax.add(lax._const(x, eps), x), lax.sub(lax._const(x, 1), x)))


def pareto(key, b, shape=None, dtype=np.float64):
  """Sample Pareto random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``b``. The default (None)
      produces a result shape equal to ``b.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``b.shape``.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `pareto` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  if shape is not None:
    shape = abstract_arrays.canonicalize_shape(shape)
  return _pareto(key, b, shape, dtype)

@partial(jit, static_argnums=(2, 3))
def _pareto(key, b, shape, dtype):
  if shape is None:
    shape = np.shape(b)
  else:
    _check_shape("pareto", shape)

  b = lax.convert_element_type(b, dtype)
  e = exponential(key, shape, dtype)
  return lax.exp(e / b)


def t(key, df, shape=(), dtype=np.float64):
  """Sample Student's t random values with given shape and float dtype.

  Args:
    key: a PRNGKey used as the random key.
    df: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``df``. The default (None)
      produces a result shape equal to ``df.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``df.shape``.
  """
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `t` must be a float "
                     f"dtype, got {dtype}")
  dtype = dtypes.canonicalize_dtype(dtype)
  shape = abstract_arrays.canonicalize_shape(shape)
  return _t(key, df, shape, dtype)

@partial(jit, static_argnums=(2, 3))
def _t(key, df, shape, dtype):
  if shape is None:
    shape = np.shape(df)
  else:
    _check_shape("t", shape, np.shape(df))

  df = lax.convert_element_type(df, dtype)
  key_n, key_g = split(key)
  n = normal(key_n, shape, dtype)
  two = _constant_like(n, 2)
  half_df = lax.div(df, two)
  g = gamma(key_n, half_df, shape, dtype)
  return n * jnp.sqrt(half_df / g)
