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

"""Interface and utility functions to XLA.

This module wraps the XLA client(s) and builders to standardize their interfaces
and provide some automatic type mapping logic for converting between Numpy and
XLA. There are also a handful of related casting utilities.
"""


from functools import partial
import os
from typing import Callable, Dict, Tuple, Union
import warnings

from absl import logging

from ..config import flags
from .. import util
from .. import dtypes
import numpy as onp  # 'onp' rather than 'np' to distinguish from autograd.numpy
import threading

try:
  from . import tpu_client
except ImportError:
  tpu_client = None
from . import xla_client

xops = xla_client.ops

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'jax_xla_backend', 'xla',
    'Default is "xla" for the XLA service directly, '
    'or "tpu_driver" for using high-performance access to Cloud TPU hardware.')
flags.DEFINE_string(
    'jax_backend_target', 'local',
    'Either "local" or "rpc:address" to connect to a remote service target.')
flags.DEFINE_string(
    'jax_platform_name',
    os.getenv('JAX_PLATFORM_NAME', ''),
    'Platform name for XLA. The default is to attempt to use a GPU if '
    'available, but fall back to CPU otherwise. To set the platform manually, '
    'pass "cpu" for CPU or "gpu" for GPU.')
flags.DEFINE_bool(
    'jax_disable_most_optimizations', False,
    'Try not to do much optimization work. This can be useful if the cost of '
    'optimization is greater than that of running a less-optimized program.')


def get_compile_options(num_replicas, num_partitions, device_assignment=None,
                        use_spmd_partitioning=True):
  """Returns the compile options to use, as derived from flag values.

  Args:
    num_replicas: int indicating the number of replicas for which to compile.
    num_partitions: int indicating the number of partitions for which to compile.
    device_assignment: Optional tuple of integers indicating the assignment of
      logical replicas to physical devices (default inherited from
      xla_client.CompileOptions). Must be consistent with `num_replicas` and
      `num_partitions`.
    use_spmd_partitioning: boolean indicating whether to enable SPMD or MPMD
      partitioning in XLA.
  """
  compile_options = xla_client.CompileOptions()
  compile_options.num_replicas = num_replicas
  compile_options.num_partitions = num_partitions
  build_options = compile_options.executable_build_options
  build_options.use_spmd_partitioning = use_spmd_partitioning
  if device_assignment is not None:
    logging.vlog(
        2,
        'get_compile_options: num_replicas=%s num_partitions=%s device_assignment=%s',
        num_replicas, num_partitions, device_assignment)
    device_assignment = onp.array(device_assignment)

    # Allow 1D device assignment if num_partitions is 1.
    if (device_assignment.ndim == 1) and (num_partitions == 1):
      device_assignment = device_assignment[:, None]

    if num_replicas != device_assignment.shape[0]:
      msg = 'device_assignment does not match num_replicas: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_replicas))

    if num_partitions != device_assignment.shape[1]:
      msg = 'device_assignment does not match num_partitions: {} vs {}.'
      raise ValueError(msg.format(device_assignment, num_partitions))

    device_assignment = xla_client.DeviceAssignment.create(device_assignment)
    assert device_assignment.replica_count() == num_replicas
    assert device_assignment.computation_count() == num_partitions
    compile_options.device_assignment = device_assignment

  if FLAGS.jax_disable_most_optimizations:
    debug_options = compile_options.executable_build_options.debug_options
    debug_options.xla_backend_optimization_level = 0
    debug_options.xla_llvm_disable_expensive_passes = True
    debug_options.xla_test_all_input_layouts = False

  return compile_options

_backends = {}

def register_backend(name, factory):
  _backends[name] = factory

def _get_local_backend(platform=None):
  if not platform:
    platform = FLAGS.jax_platform_name or None

  backend = xla_client.get_local_backend(platform)
  if backend is None:
    raise RuntimeError("No local XLA backends found.")

  if backend.platform == 'cpu' and platform != 'cpu':
    warnings.warn('No GPU/TPU found, falling back to CPU.')

  return backend


register_backend('xla', _get_local_backend)

# memoize the TPU driver to be consistent with xla_client behavior
_tpu_backend = None

def _get_tpu_driver_backend(platform):
  del platform
  global _tpu_backend
  if _tpu_backend is None:
    backend_target = FLAGS.jax_backend_target
    if backend_target is None:
      raise ValueError('When using TPU Driver as the backend, you must specify '
                       '--jax_backend_target=<hostname>:8470.')
    _tpu_backend = tpu_client.TpuBackend.create(worker=backend_target)
  return _tpu_backend


if tpu_client:
  register_backend('tpu_driver', _get_tpu_driver_backend)


_backend_lock = threading.Lock()

@util.memoize
def get_backend(platform=None):
  # TODO(mattjj,skyewm): remove this input polymorphism after we clean up how
  # 'backend' values are handled
  if not isinstance(platform, (type(None), str)):
    return platform

  with _backend_lock:
    backend = _backends.get(FLAGS.jax_xla_backend)
    if backend is None:
      msg = 'Unknown jax_xla_backend value "{}".'
      raise ValueError(msg.format(FLAGS.jax_xla_backend))
    return backend(platform)


def get_device_backend(device=None):
  """Returns the Backend associated with `device`, or the default Backend."""
  platform = device.platform if device else None
  return get_backend(platform)


def device_count(backend: str = None):
  """Returns the total number of devices.

  On most platforms, this is the same as :py:func:`jax.local_device_count`.
  However, on multi-host platforms, this will return the total number of devices
  across all hosts.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    Number of devices.
  """
  return int(get_backend(backend).device_count())


def local_device_count(backend: str =None):
  """Returns the number of devices on this host."""
  return int(get_backend(backend).local_device_count())


def devices(backend: str = None):
  """Returns a list of all devices for a given backend.

  Each device is represented by a subclass of :class:`Device` (e.g.
  :class:`CpuDevice`, :class:`GpuDevice`). The length of the returned list is
  equal to ``device_count(backend)``. Local devices can be identified by comparing
  :meth:`Device.host_id` to the value returned by :py:func:`jax.host_id`.

  If ``backend`` is ``None``, returns all the devices from the default backend.
  The default backend is generally ``'gpu'`` or ``'tpu'`` if available,
  otherwise ``'cpu'``.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of Device subclasses.
  """
  return get_backend(backend).devices()


def local_devices(host_id: int = None, backend: str = None):
  """Like :py:func:`jax.devices`, but only returns devices local to a given host.

  If ``host_id`` is ``None``, returns devices local to this host.

  Args:
    host_id: the integer ID of the host. Host IDs can be retrieved via
      :py:func:`jax.host_ids`.
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    List of Device subclasses.
  """
  if host_id is None:
    host_id = get_backend(backend).host_id()
  if host_id not in host_ids():
    raise ValueError(f"Unknown host_id {host_id}")
  return [d for d in devices(backend) if d.host_id == host_id]


def host_id(backend: str = None):
  """Returns the integer host ID of this host.

  On most platforms, this will always be 0. This will vary on multi-host
  platforms though.

  Args:
    backend: This is an experimental feature and the API is likely to change.
      Optional, a string representing the xla backend: ``'cpu'``, ``'gpu'``, or
      ``'tpu'``.

  Returns:
    Integer host ID.
  """
  return get_backend(backend).host_id()


def host_ids(backend: str = None):
  """Returns a sorted list of all host IDs."""
  return sorted(list(set(d.host_id for d in devices(backend))))


def host_count(backend: str = None):
  """Returns the number of hosts."""
  return len(host_ids(backend))


### utility functions

@util.memoize
def dtype_to_etype(dtype):
  """Convert from dtype to canonical etype (reading FLAGS.jax_enable_x64)."""
  return xla_client.dtype_to_etype(dtypes.canonicalize_dtype(dtype))


@util.memoize
def supported_numpy_dtypes():
  return {dtypes.canonicalize_dtype(dtype)
          for dtype in xla_client.XLA_ELEMENT_TYPE_TO_DTYPE.values()}


# TODO(mattjj,frostig): try to remove this function
def normalize_to_xla_dtypes(val):
  """Normalize dtypes in a value."""
  if hasattr(val, '__array__') or onp.isscalar(val):
    return onp.asarray(val,
                       dtype=dtypes.canonicalize_dtype(dtypes.result_type(val)))
  elif isinstance(val, (tuple, list)):
    return tuple(normalize_to_xla_dtypes(x) for x in val)
  raise TypeError('Can\'t convert to XLA: {}'.format(val))

def _numpy_array_constant(builder, value, canonicalize_types=True):
  if canonicalize_types:
    value = normalize_to_xla_dtypes(value)
  return xops.ConstantLiteral(builder, value)

def parameter(builder, num, shape, name=None, replicated=None):
  if name is None:
    name = ''
  if replicated is None:
    replicated = []
  elif isinstance(replicated, bool):
    replicated = [replicated] * shape.leaf_count()

  return xops.Parameter(builder, num,
                        shape.with_major_to_minor_layout_if_absent(), name,
                        replicated)


def constant(builder, py_val, canonicalize_types=True):
  """Translate constant `py_val` to a constant, canonicalizing its dtype.

  Args:
    py_val: a Python value to be translated to a constant.

  Returns:
    A representation of the constant, either a ComputationDataHandle or None
  """
  py_type = type(py_val)
  if py_type in _constant_handlers:
    return _constant_handlers[py_type](builder, py_val, canonicalize_types)
  else:
    raise TypeError("No constant handler for type: {}".format(py_type))

# HLO instructions optionally can be annotated to say how the output should be
# spatially partitioned (represented in XLA as OpSharding protos, see
# _sharding_to_proto). For array outputs, the annotation is either an int per
# dimension specifying the number of ways that dimension divided (i.e. the total
# number of shards is the product), or None to indicate the array should be
# replicated. Tuple outputs are represented as tuples thereof. XLA supports
# arbitrary tuple nesting, but JAX only uses one level of tupling (and our type
# checkers don't support recursive types), so we only represent one level of
# nesting in this type definition.
SpatialSharding = Union[Tuple[int, ...],
                        None,
                        Tuple[Union[Tuple[int, ...], None], ...]]

def _sharding_to_proto(sharding: SpatialSharding):
  """Converts a SpatialSharding to an OpSharding.

  See
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/xla_data.proto#L601
  for details on the OpSharding proto.
  """
  proto = xla_client.OpSharding()
  if isinstance(sharding, tuple) and not isinstance(sharding[0], int):
      assert all(s is None or isinstance(s, tuple) for s in sharding)
      sub_protos = [_sharding_to_proto(s) for s in sharding]  # type: ignore
      proto.type = xla_client.OpSharding.Type.TUPLE
      proto.tuple_shardings = sub_protos
      return proto

  if sharding is None:
    proto.type = xla_client.OpSharding.Type.REPLICATED
  else:
    proto.type = xla_client.OpSharding.Type.OTHER
    proto.tile_assignment_dimensions = list(sharding)
    proto.tile_assignment_devices = list(range(onp.product(sharding)))
  return proto

def set_sharding(builder, op, sharding: SpatialSharding):
  """Uses CustomCall to annotate a value as sharded."""
  # "Sharding" is a built-in custom call target that acts like an identity
  # function, and is used to attach an OpSharding to.
  return with_sharding(builder, sharding, xops.CustomCall,
                       builder, b"Sharding", [op], builder.get_shape(op))

def with_sharding(builder, sharding: SpatialSharding, op_fn, *args, **kwargs):
  """Builds op_fn(*args, **kwargs) with sharding annotation."""
  builder.set_sharding(_sharding_to_proto(sharding))
  try:
    return op_fn(*args, **kwargs)
  finally:
    builder.clear_sharding()

def make_computation_builder(name):
  return xla_client.XlaBuilder(name)


def register_constant_handler(type_, handler_fun):
  _constant_handlers[type_] = handler_fun
_constant_handlers: Dict[type, Callable] = {}


def _ndarray_constant_handler(c, val, canonicalize_types=True):
  """Constant handler for ndarray literals, handling zero-size strides.

  This function essentially calls _numpy_array_constant(val) except it has
  special handling of arrays with any strides of size zero: for those, it
  generates appropriate calls to NumpyArrayConstant, Broadcast, and Transpose
  to avoid staging in large literals that might arise from np.zeros or np.ones
  or the output of lax.broadcast (which uses onp.broadcast_to which in turn
  uses size-zero strides).

  Args:
    c: an XlaBuilder
    val: an ndarray.

  Returns:
    An XLA ComputationDataHandle / XlaOp representing the constant ndarray
    staged into the XLA Computation.
  """
  # TODO(mattjj): revise this to use xops.BroadcastInDim rather than Transpose
  if onp.any(onp.equal(0, val.strides)) and val.size > 0:
    zero_stride_axes, = onp.where(onp.equal(0, val.strides))
    other_axes, = onp.where(onp.not_equal(0, val.strides))
    collapsed_val = val[tuple(0 if ax in zero_stride_axes else slice(None)
                              for ax in range(val.ndim))]
    xla_val = xops.Broadcast(
        _numpy_array_constant(c, collapsed_val, canonicalize_types),
        onp.take(val.shape, zero_stride_axes))
    permutation = onp.argsort(tuple(zero_stride_axes) + tuple(other_axes))
    return xops.Transpose(xla_val, permutation)
  else:
    return _numpy_array_constant(c, val, canonicalize_types)
register_constant_handler(onp.ndarray, _ndarray_constant_handler)


def _scalar_constant_handler(c, val, canonicalize_types=True):
  return _numpy_array_constant(c, val, canonicalize_types)

for scalar_type in [onp.int8, onp.int16, onp.int32, onp.int64,
                    onp.uint8, onp.uint16, onp.uint32, onp.uint64,
                    onp.float16, onp.float32, onp.float64, onp.float128,
                    onp.bool_, onp.longlong]:
  register_constant_handler(scalar_type, _scalar_constant_handler)

def _python_scalar_handler(dtype, c, val, canonicalize_dtypes=True):
  return _numpy_array_constant(c, dtype.type(val))

for ptype, dtype in dtypes.python_scalar_dtypes.items():
  register_constant_handler(ptype, partial(_python_scalar_handler, dtype))
