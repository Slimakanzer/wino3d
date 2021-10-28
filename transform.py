import numpy as np


def assert_array_equal(a, b):
    assert np.allclose(a, b)


def hadamard_product(a, b):
    assert a.shape == b.shape

    result = np.zeros(a.shape)
    for idx, x in np.ndenumerate(result):
        result[idx] = a[idx] * b[idx]

    return result


def nmod_tensor_product(A, U, n):
    """
    n-Mode Tensor - Matrix Product
    ref: https://arxiv.org/pdf/1611.06565.pdf

    Input:
    A: tensor of R^( I_1 × I_2 × .. I_n × .. I_N )
    U: matrix of R^( J × I_n)
    n: scalar within [1:N], specifying the mode

    Output:
    B: output tensor of R^(I_1 × I_2 × .. J × .. I_N)

    Calculation:
    B[i1,i2,..,j,..,iN] = sum(A[i1,i2,..,in,..,iN] * U[j,in]) by axis 'in'
    """
    assert n > 0 and n <= len(A.shape)
    assert A.shape[n-1] == U.shape[1]

    i_n = A.shape[n-1]
    j = U.shape[0]

    result_shape = np.asarray(A.shape)
    result_shape[n-1] = j
    result = np.zeros(result_shape)

    for idx, x in np.ndenumerate(result):
        j_idx = idx[n-1]

        acc = 0.0
        for i in range(i_n):
            input_idx = np.asarray(idx)
            input_idx[n-1] = i
            acc += A[tuple(input_idx)] * U[j_idx, i]

        result[idx] = acc

    return result


X = np.matrix(np.random.rand(4, 4))
d = np.matrix(np.random.rand(4, 4))

assert_array_equal(nmod_tensor_product(d, X, 1),
                   X * d,)
assert_array_equal(nmod_tensor_product(d, X, 2),
                   d * X.T,)
assert_array_equal(nmod_tensor_product(nmod_tensor_product(d, X, 1), X, 2),
                   X * d * X.T,)

#---------------------------------------------------------------------#
#---------------3D convolution using Winograd algorithm---------------#
#---------------------------------------------------------------------#


def conv_ref(data, filter, out):
    for o_idx, _ in np.ndenumerate(out):
        d_idx_x = o_idx[0]
        d_idx_y = o_idx[1]
        d_idx_z = o_idx[2]

        acc = 0.0
        for f_idx, _ in np.ndenumerate(filter):
            d_idx = ((d_idx_x+f_idx[0]),
                     (d_idx_y+f_idx[1]),
                     (d_idx_z+f_idx[2]))

            acc += data[d_idx]*filter[f_idx]

        out[o_idx] = acc
    return out


def conv_wino(data, filter, BT, G, AT):
    d_transform = nmod_tensor_product(
        nmod_tensor_product(
            nmod_tensor_product(d, BT, 1),
            BT, 2),
        BT, 3)

    f_transform = nmod_tensor_product(
        nmod_tensor_product(
            nmod_tensor_product(f, G, 1),
            G, 2),
        G, 3)

    accum = hadamard_product(d_transform, f_transform)
    x_back_transform = nmod_tensor_product(accum, AT, 1)
    y_back_transform = nmod_tensor_product(x_back_transform, AT, 2)
    z_back_transform = nmod_tensor_product(y_back_transform, AT, 3)
    return z_back_transform


# F(2,3) transform
out_tile = 2
filter_tile = 3
data_tile = out_tile + filter_tile - 1

BT = np.array([[1,  0, -1,  0],
               [0,  1,  1,  0],
               [0, -1,  1,  0],
               [0,  1,  0, -1]])

G = np.array([[1,      0,   0],
              [0.5,  0.5, 0.5],
              [0.5, -0.5, 0.5],
              [0,      0,   1]])

AT = np.array([[1, 1,  1,  0],
               [0, 1, -1, -1]])

d = np.array(np.arange(0, data_tile**3)
             .reshape(data_tile, data_tile, data_tile))

f = np.array(np.arange(0, filter_tile**3)
             .reshape(filter_tile, filter_tile, filter_tile))

o = np.zeros((out_tile, out_tile, out_tile))

conv_wino = conv_wino(d, f, BT, G, AT)
conv_ref = conv_ref(d, f, o)

assert_array_equal(conv_wino, conv_ref)
print(conv_wino)
