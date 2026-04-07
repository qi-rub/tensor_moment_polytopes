import numpy as np

def apply_system(g_i, T, system):
    """ Apply the linear map `g_i` on the system `system` of tensor `T`. """
    k = T.ndim
    t = list(range(k))
    t[system], t[k-2] = t[k-2], t[system]
    return (np.array(g_i, dtype='O') @ T.transpose(t)).transpose(t)

def apply(g, T):
    for i, g_i in enumerate(g):
        T = apply_system(g_i, T, i)
    return T

# Define the tensors.
# Return the elementary tensors of shape 3x3x3.
e = lambda i,j,k: reduce(lambda x,y: np.tensordot(x, y, axes=0), np.eye(3, dtype=int)[[i,j,k]])

Tdet = e(0,1,2) - e(0,2,1) - e(1,0,2) + e(1,2,0) + e(2,0,1) - e(2,1,0)
W = e(0,0,1) + e(0,1,0) + e(1,0,0)
T10 = e(0,0,2) + e(0,1,1) + e(0,2,0) + e(1,0,1) + e(1,1,0) + e(2,0,0)
T6 = e(0,0,2) + e(0,1,1) + e(1,0,1) + e(1,2,0) + e(2,1,0)
T17 = e(0,0,1) + e(0,1,0) + e(1,0,2) + e(1,2,0)

Tdet111_to_T10 = [
    Matrix([
        [x^(-1),       0, -x^(-1)],
        [     0,       1,       0],
        [     0,       0,      -x],
    ]),
    Matrix([
        [x^(-1),  0,  0],
        [     0, -1, -1],
        [     0,  0,  x],
    ]),
    Matrix([
        [ -1, -1,   0],
        [  0,  0,  -x],
        [x^2,  0,   0],
    ])]

TdetW_to_T6 = [
    Matrix([
        [  x^(-1), -1/3*x^(-1),         0],
        [       0, -1/3*x^(-2),  -x^(-2)],
        [       0,            1,         0],
    ]),
    Matrix([
        [  -x, -1/3*x,    0],
        [   0,   -1/3,   -1],
        [   0,  -x^2,    0],
    ]),
    Matrix([
        [-1,    0,  0],
        [ 0,    x,  0],
        [ 0, -1/3, -1],
    ])]

Tdet_to_T17 = [
    Matrix([
        [ 0,  0, -1],
        [-1,  0,  0],
        [ 0,  0,  0]
    ]),
    Matrix([
        [ 0,  1,  0],
        [ 1,  0,  0],
        [ 0,  0, -1],
    ]),
    Matrix([
        [ 0, -1,  0],
        [ 1,  0,  0],
        [ 0,  0, -1],
    ])]

print("Degeneration from Tdet + W to T6:     ",
      vector(apply(TdetW_to_T6, Tdet + W).flatten()).subs({x: 0}) == vector(T6.flatten()))
print("Degeneration from Tdet + e111 to T10: ",
      vector(apply(Tdet111_to_T10, Tdet + e(0,0,0)).flatten()).subs({x: 0}) == vector(T10.flatten()))
print("Degeneration from Tdet to T17:        ",
      np.all(apply(Tdet_to_T17, Tdet) == T17))
