import numpy as np

def axis_slice(axis, slice_target, ndim):
    slice_none = [slice(None)]

    if axis == 0:
        s = slice_target + slice_none * (ndim - 1)
    elif axis == ndim - 1:
        s = slice_none * (ndim - 1) + slice_target
    else:
        s = slice_none * (axis) + slice_target + slice_none * (ndim - axis - 1)

    s = tuple(s)

    return s

def padding_edge_reflect(x: np.ndarray, padwidth: int, axis = None) -> np.ndarray:
    axis = x.ndim - 1 if (axis is None or axis < 0) else axis

    dn = x.shape[axis]

    if padwidth < 0 or padwidth > dn - 1:
        raise ValueError('padwidth')

    s = axis_slice(axis, [slice(1, padwidth + 1)], x.ndim)
    x_edge_a = x[s]

    s = axis_slice(axis, [slice(0, 1)], x.ndim)
    xa = x[s]

    x_reflact_a = 2 * xa - np.flip(x_edge_a, axis)

    s = axis_slice(axis, [slice(-padwidth - 1, -1)], x.ndim)
    x_edge_b = x[s]

    s = axis_slice(axis, [slice(-1, None)], x.ndim)
    xb = x[s]

    x_reflact_b = 2 * xb - np.flip(x_edge_b, axis)

    y = np.concatenate([x_reflact_a, x, x_reflact_b], axis)
    
    return y
