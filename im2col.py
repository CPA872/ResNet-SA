# Partitionally based on Stanford CS231N assignment https://cs231n.github.io/assignments2023/assignment2/

import numpy as np


def pair(x):
    return (x, x)


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    padding = pair(padding) if isinstance(padding, int) else padding
    stride  = pair(stride) if isinstance(stride, int) else stride

    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    # assert (H + 2 * padding[0] - field_height) % stride[0] == 0
    # assert (W + 2 * padding[1] - field_height) % stride[1] == 0
    out_height = int((H + 2 * padding[0] - field_height) / stride[0] + 1)
    out_width = int((W + 2 * padding[1] - field_width) / stride[1] + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = pair(padding) if isinstance(padding, int) else padding
    x_padded = np.pad(x, ((0, 0), (0, 0), p, p), mode="constant")

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0)
    cols = cols.reshape(field_height * field_width * C, -1)
    
    return cols
