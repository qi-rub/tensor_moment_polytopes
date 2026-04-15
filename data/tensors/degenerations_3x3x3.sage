# Explicit degenerations in Nurmiev's 3x3x3 tensor classification.
#
# Tensor naming/numbering follows Nurmiev (Orbits and invariants of cubic matrices of order three, 2000).
#
# A degeneration S >= T means:
#   lim_{x->0} (A(x) x B(x) x C(x)) . S = T
# where A(x), B(x), C(x) are invertible matrices with Laurent polynomial entries.
#
# Cyclic permutations of the tensors are denoted by pi(T, p):
#   pi(T, 1): T[i,j,k] -> T[k,i,j]
#   pi(T, 2): T[i,j,k] -> T[j,k,i]

import numpy as np
from functools import reduce

def apply_system(g_i, T, system):
    """Apply the linear map g_i on the system of tensor T."""
    k = T.ndim
    t = list(range(k))
    t[system], t[k-2] = t[k-2], t[system]
    return (np.array(g_i, dtype='O') @ T.transpose(t)).transpose(t)

def apply(g, T):
    """Apply the tuple of linear maps g to the tensor T."""
    for i, g_i in enumerate(g):
        T = apply_system(g_i, T, i)
    return T

def pi(T, p):
    """Cyclic permutation of tensor legs. pi^1: T[i,j,k] -> T[k,i,j]."""
    if p == 0: return T
    if p == 1: return np.transpose(T, (2,0,1))
    if p == 2: return np.transpose(T, (1,2,0))

e = lambda i,j,k: reduce(lambda x,y: np.tensordot(x, y, axes=0), np.eye(3, dtype=int)[[i,j,k]])

# Unstable tensors.
T1 = e(0,1,2) + e(0,2,1) + e(1,0,2) + e(1,1,1) + e(1,2,0) + e(2,0,0)
T2 = e(0,1,2) + e(0,2,1) + e(1,0,2) + e(1,1,0) + e(1,1,1) + e(2,0,0)
T3 = e(0,0,2) + e(0,1,1) + e(0,2,0) + e(1,0,1) + e(1,1,2) + e(2,0,0)
T4 = e(0,0,2) + e(0,1,1) + e(1,0,1) + e(1,1,0) + e(2,2,0)
T5 = e(0,0,2) + e(0,2,0) + e(0,2,1) + e(1,1,0) + e(2,0,1)
T6 = e(0,0,2) + e(0,1,1) + e(1,0,1) + e(1,2,0) + e(2,1,0)
T7 = e(0,0,2) + e(0,1,1) + e(0,2,0) + e(1,0,1) + e(2,1,0)
T8 = e(0,0,2) + e(0,2,0) + e(1,1,1) + e(2,0,0)
T9 = e(0,0,0) + e(0,1,1) + e(1,1,1) + e(1,2,2)
T10 = e(0,0,2) + e(0,1,1) + e(0,2,0) + e(1,0,1) + e(1,1,0) + e(2,0,0)
T11 = e(0,0,2) + e(0,2,0) + e(1,0,1) + e(2,1,0)
T12 = e(0,0,2) + e(0,2,0) + e(1,0,0) + e(1,1,1)
T13 = e(0,0,2) + e(0,1,1) + e(0,2,0) + e(1,0,1) + e(1,1,0)
T14 = e(0,0,2) + e(0,1,0) + e(0,2,1) + e(1,0,0) + e(2,0,1)
T15 = e(0,1,1) + e(0,2,2) + e(1,0,0)
T16 = e(0,0,2) + e(0,1,1) + e(0,2,0) + e(1,0,0)
T17 = e(0,0,1) + e(0,1,0) + e(1,0,2) + e(1,2,0)
T18 = e(0,0,0) + e(0,1,1) + e(1,0,1) + e(1,1,2)
T19 = e(0,0,2) + e(0,1,0) + e(1,0,1)
T20 = e(0,0,0) + e(1,1,1)
T21 = e(0,0,1) + e(0,1,0) + e(1,0,0)
T22 = e(0,0,0) + e(0,1,1) + e(0,2,2)
T23 = e(0,0,0) + e(0,1,1)
T24 = e(0,0,0)
T25 = 0*e(0,0,0)

# Family 4.
Tdet = e(0,1,2) - e(0,2,1) - e(1,0,2) + e(1,2,0) + e(2,0,1) - e(2,1,0)
W = e(0,0,1) + e(0,1,0) + e(1,0,0)

# T1 >= T2, d=(8,6,4)
T1_to_T2 = [
    Matrix([
        [x, -1/3*x^(-2), 1/27*x^(-8)],
        [0,           x, -1/3*x^(-5)],
        [0,           0,      x^(-2)]
    ]),
    Matrix([
        [1, 1/3*x^(-3), 1/9*x^(-6)],
        [0,          1, 2/3*x^(-3)],
        [0,          0,          1]
    ]),
    Matrix([
        [x^2, 1/3/x,  1/9*x^(-4)],
        [  0,   1/x, -1/3*x^(-4)],
        [  0,     0,         1/x]
    ])
]

# T2 >= T3, d=(1,0,0)
T2_to_T3 = [
    Matrix([
        [-1/x,   0, -1/x],
        [   0, 1/x,    0],
        [  -1,   0,    0]
    ]),
    Matrix([
        [ 1,  0, 1],
        [ 0, -1, 0],
        [-x,  0, 0]
    ]),
    Matrix([
        [ 1, -1, 0],
        [ 0,  0, x],
        [-x,  0, 0]
    ])
]

# T2 >= pi(T4,1), d=(1,0,0)
T2_to_T4 = [
    Matrix([
        [-1/x,    0,    0],
        [   0, -1/x,    0],
        [   0,    0, -1/x]
    ]),
    Matrix([
        [-x,  0,  0],
        [ 0, -x,  0],
        [ 0,  0, -1]
    ]),
    Matrix([
        [1, 0, 0],
        [0, 0, 1],
        [0, x, 0]
    ])
]

# T3 >= pi(T5,1), d=(1,0,0)
T3_to_T5 = [
    Matrix([
        [     0,   1/x,      0],
        [-1/2/x, 1/2/x, -1/2/x],
        [     1,     0,      0]
    ]),
    Matrix([
        [1/2, -1/2, -1/2],
        [  0,    x,    0],
        [  0,    0,    x]
    ]),
    Matrix([
        [-2, 0,  0],
        [ 0, 1,  1],
        [ 0, x, -x]
    ])
]

# T3 >= T6, d=(1,0,0)
T3_to_T6 = [
    Matrix([
        [0, -1/x,    0],
        [1,    0,    0],
        [0,    0, -1/x]
    ]),
    Matrix([
        [ 0, 1, 0],
        [-x, 0, 0],
        [ 0, 0, 1]
    ]),
    Matrix([
        [1, 0,  0],
        [0, 1,  0],
        [0, 0, -x]
    ])
]

# T4 >= T5, d=(1,0,0)
T4_to_T5 = [
    Matrix([
        [-1/x, -1/x,    0],
        [   0,    0, -1/x],
        [  -1,    0,    0]
    ]),
    Matrix([
        [-1, 0, 0],
        [ 0, 0, x],
        [ 0, x, 0]
    ]),
    Matrix([
        [-1,  0, 0],
        [ 0, -1, 1],
        [ 0,  x, 0]
    ])
]

# T4 >= T6, d=(1,1,0)
T4_to_T6 = [
    Matrix([
        [-1,  0,  0],
        [ 0, -1, -1],
        [ 0,  0, -x]
    ]),
    Matrix([
        [1/x,   0,    0],
        [  0, 1/x, -1/x],
        [  0,  -1,    0]
    ]),
    Matrix([
        [1,  0,  0],
        [0, -x,  0],
        [0,  0, -x]
    ])
]

# T5 >= T7, d=(1,0,0)
T5_to_T7 = [
    Matrix([
        [1/x, -1/x,    0],
        [  0,    0, -1/x],
        [ -1,    0,    0]
    ]),
    Matrix([
        [1,  0,  0],
        [0, -1, -1],
        [0, -x,  0]
    ]),
    Matrix([
        [1,  0, 0],
        [0, -x, 0],
        [0,  0, x]
    ])
]

# T5 >= T8, d=(1,0,0)
T5_to_T8 = [
    Matrix([
        [-1,    0,  0],
        [ 0, -1/x,  0],
        [ 0,    0, -1]
    ]),
    Matrix([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1]
    ]),
    Matrix([
        [0, 1, 0],
        [x, 0, 0],
        [0, 0, 1]
    ])
]

# T5 >= pi(T9,2), restriction
T5_to_T9 = [
    Matrix([
        [ 0,  0, 1],
        [-1,  0, 0],
        [ 0, -1, 0]
    ]),
    Matrix([
        [-1, 0, 0],
        [ 0, 0, 1],
        [ 0, 1, 0]
    ]),
    Matrix([
        [ 0, -1, 0],
        [-1,  0, 0],
        [ 0,  0, 0]
    ])
]

# T6 >= T7, d=(1,0,0)
T6_to_T7 = [
    Matrix([
        [-1/x, -1/x,    0],
        [   0,    1,    0],
        [   0,    0, -1/x]
    ]),
    Matrix([
        [-1, 0, 0],
        [ 0, x, 0],
        [ 0, 0, x]
    ]),
    Matrix([
        [-1,  0, 0],
        [ 0, -1, 1],
        [ 0,  0, x]
    ])
]

# T7 >= T10, d=(0,0,2)
T7_to_T10 = [
    Matrix([
        [1, 0,   0],
        [0, 1,   x],
        [0, 0, x^2]
    ]),
    Matrix([
        [ -x, -1,   -1],
        [x^2,  0,    0],
        [  0,  0, -x^2]
    ]),
    Matrix([
        [-x^(-2), x^(-2),      0],
        [      0,   -1/x, x^(-2)],
        [      0,     -1,      0]
    ])
]

# T7 >= T11, d=(1,0,0)
T7_to_T11 = [
    Matrix([
        [1, 0,   0],
        [0, 1,   0],
        [0, 0, 1/x]
    ]),
    Matrix([
        [1,  0,  0],
        [0, -x,  0],
        [0,  0, -1]
    ]),
    Matrix([
        [-1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 1]
    ])
]

# T7 >= pi(T12,1), restriction
T7_to_T12 = [
    Matrix([
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0]
    ]),
    Matrix([
        [-1,  0, 0],
        [ 0, -1, 0],
        [ 0,  0, 0]
    ]),
    Matrix([
        [ 0, -1,  0],
        [-1,  0,  0],
        [ 0,  0, -1]
    ])
]

# T8 >= T10, d=(0,0,2)
T8_to_T10 = [
    Matrix([
        [   1,  1,  0],
        [   0, -x, -x],
        [-x^2,  0,  0]
    ]),
    Matrix([
        [1,   1,  0],
        [x,   0, -x],
        [0, x^2,  0]
    ]),
    Matrix([
        [-x^(-2), x^(-2), -x^(-2)],
        [   -1/x,      0,       0],
        [      0,      1,       0]
    ])
]

# T8 >= T12, restriction
T8_to_T12 = [
    Matrix([
        [-1, 0,  0],
        [ 0, 1, -1],
        [ 0, 0,  0]
    ]),
    Matrix([
        [1,  0, 0],
        [0, -1, 0],
        [0,  0, 1]
    ]),
    Matrix([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1]
    ])
]

# T9 >= T12, d=(0,1,0)
T9_to_T12 = [
    Matrix([
        [-1, 0, 0],
        [ 0, x, 0],
        [ 0, 0, 0]
    ]),
    Matrix([
        [-1/x, 1/x,   0],
        [   0,   0, 1/x],
        [  -1,   0,   0]
    ]),
    Matrix([
        [1,  1, 0],
        [0,  0, 1],
        [0, -x, 0]
    ])
]

# T10 >= T13, restriction
T10_to_T13 = [
    Matrix([
        [-1,  0, 0],
        [ 0, -1, 0],
        [ 0,  0, 0]
    ]),
    Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ]),
    Matrix([
        [-1,  0,  0],
        [ 0, -1,  0],
        [ 0,  0, -1]
    ])
]

# T10 >= T14, d=(1,0,0)
T10_to_T14 = [
    Matrix([
        [1/x, 0, 0],
        [  0, 0, 1],
        [  0, 1, 0]
    ]),
    Matrix([
        [1, 0, 0],
        [0, 0, x],
        [0, x, 0]
    ]),
    Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, x]
    ])
]

# T11 >= pi(T14,1), d=(1,0,0)
T11_to_T14 = [
    Matrix([
        [1/x,    0,    0],
        [  0, -1/x, -1/x],
        [  0,    0,    1]
    ]),
    Matrix([
        [-1, -1,  0],
        [ 0,  0, -x],
        [ 0,  x,  0]
    ]),
    Matrix([
        [-1, 1,  0],
        [ 0, 0, -x],
        [ x, 0,  0]
    ])
]

# T11 >= pi(T15,1), restriction
T11_to_T15 = [
    Matrix([
        [ 0, 0, 1],
        [-1, 0, 0],
        [ 0, 1, 0]
    ]),
    Matrix([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    Matrix([
        [1, 0,  0],
        [0, 0, -1],
        [0, 1,  0]
    ])
]

# T12 >= T13, d=(2,1,0)
T12_to_T13 = [
    Matrix([
        [-x^(-2), 1/2*x^(-2), 0],
        [      0,       -1/x, 0],
        [      0,          0, 0]
    ]),
    Matrix([
        [1/2, -2/x, -1/4],
        [  x,    0,    0],
        [  0,    0,  x^2]
    ]),
    Matrix([
        [   -1,  -1/4*x, -1/2],
        [    0, 1/2*x^2,   -x],
        [2*x^2,       0,    0]
    ])
]

# T12 >= T15, d=(1,0,0)
T12_to_T15 = [
    Matrix([
        [-1/x,   0, 0],
        [   0, 1/x, 0],
        [   0,   0, 0]
    ]),
    Matrix([
        [ 0, -x,  0],
        [ 0,  0, -1],
        [-x,  0,  0]
    ]),
    Matrix([
        [0, -1, 0],
        [x,  0, 0],
        [0,  0, 1]
    ])
]

# T13 >= T16, d=(1,0,0)
T13_to_T16 = [
    Matrix([
        [-1/x, 0, 0],
        [   0, 1, 0],
        [   0, 0, 0]
    ]),
    Matrix([
        [-1, 0,  0],
        [ 0, 0, -1],
        [ 0, x,  0]
    ]),
    Matrix([
        [0, -1, 0],
        [x,  0, 0],
        [0,  0, x]
    ])
]

# T13 >= T17, d=(1,0,0)
T13_to_T17 = [
    Matrix([
        [   0, 1/x, 0],
        [-1/x,   0, 0],
        [   0,   0, 0]
    ]),
    Matrix([
        [1, 0,  0],
        [0, x,  0],
        [0, 0, -x]
    ]),
    Matrix([
        [1, 0,  0],
        [0, x,  0],
        [0, 0, -x]
    ])
]

# T13 >= T18, restriction
T13_to_T18 = [
    Matrix([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 0]
    ]),
    Matrix([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 0]
    ]),
    Matrix([
        [-1, 0,  0],
        [ 0, 1,  0],
        [ 0, 0, -1]
    ])
]

# T14 >= T16, restriction
T14_to_T16 = [
    Matrix([
        [1, 0, 2],
        [0, 1, 0],
        [0, 0, 0]
    ]),
    Matrix([
        [-1,  0,  1],
        [ 0,  0, -1],
        [ 0, -1,  0]
    ]),
    Matrix([
        [-1,  0,  0],
        [ 0, -1,  1],
        [ 0,  0, -1]
    ])
]

# T14 >= pi(T17,2), restriction
T14_to_T17 = [
    Matrix([
        [1,  0,  0],
        [0,  0, -1],
        [0, -1,  0]
    ]),
    Matrix([
        [1,  0,  0],
        [0,  0, -1],
        [0, -1,  0]
    ]),
    Matrix([
        [ 0, -1, 0],
        [-1,  0, 0],
        [ 0,  0, 0]
    ])
]

# T15 >= T16, d=(1,0,0)
T15_to_T16 = [
    Matrix([
        [-1/x, -1/x, 0],
        [  -1,    0, 0],
        [   0,    0, 0]
    ]),
    Matrix([
        [-1,  0, -1],
        [ 0, -x,  0],
        [ 0,  0, -x]
    ]),
    Matrix([
        [-1, 0, 1],
        [ 0, 1, 0],
        [ x, 0, 0]
    ])
]

# T16 >= T19, restriction
T16_to_T19 = [
    Matrix([
        [-1, 0, 0],
        [ 0, 1, 0],
        [ 0, 0, 0]
    ]),
    Matrix([
        [-1,  0, 0],
        [ 0, -1, 0],
        [ 0,  0, 0]
    ]),
    Matrix([
        [ 0, 1, 0],
        [-1, 0, 0],
        [ 0, 0, 1]
    ])
]

# T16 >= T22, restriction
T16_to_T22 = [
    Matrix([
        [1, -1, 0],
        [0,  0, 0],
        [0,  0, 0]
    ]),
    Matrix([
        [1,  0, 1],
        [0, -1, 0],
        [0,  0, 1]
    ]),
    Matrix([
        [0,  0, 1],
        [0, -1, 0],
        [1,  0, 0]
    ])
]

# T17 >= T19, restriction
T17_to_T19 = [
    Matrix([
        [ 0, -1, 0],
        [-1,  0, 0],
        [ 0,  0, 0]
    ]),
    Matrix([
        [1, 0,  0],
        [0, 0, -1],
        [0, 0,  0]
    ]),
    Matrix([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ])
]

# T18 >= T19, d=(1,0,0)
T18_to_T19 = [
    Matrix([
        [ 0, 1/x, 0],
        [-1,   0, 0],
        [ 0,   0, 0]
    ]),
    Matrix([
        [1, 0, 0],
        [0, x, 0],
        [0, 0, 0]
    ]),
    Matrix([
        [ 0, 0, 1],
        [-1, 0, 0],
        [ 0, x, 0]
    ])
]

# T19 >= T20, restriction
T19_to_T20 = [
    Matrix([
        [0, -1, 0],
        [1,  0, 0],
        [0,  0, 0]
    ]),
    Matrix([
        [1,  0, 0],
        [0, -1, 0],
        [0,  0, 0]
    ]),
    Matrix([
        [ 0, -1, 0],
        [-1,  0, 0],
        [ 0,  0, 0]
    ])
]

# T20 >= T21, d=(1,0,0)
T20_to_T21 = [
    Matrix([
        [1/x, 1/x, 0],
        [  0,  -1, 0],
        [  0,   0, 0]
    ]),
    Matrix([
        [-1, -1, 0],
        [-x,  0, 0],
        [ 0,  0, 0]
    ]),
    Matrix([
        [-1, 1, 0],
        [-x, 0, 0],
        [ 0, 0, 0]
    ])
]

# T21 >= T23, restriction
T21_to_T23 = [
    Matrix([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]),
    Matrix([
        [ 0, -1, 0],
        [-1,  0, 0],
        [ 0,  0, 0]
    ]),
    Matrix([
        [-1,  0, 0],
        [ 0, -1, 0],
        [ 0,  0, 0]
    ])
]

# T22 >= T23, restriction
T22_to_T23 = [
    Matrix([
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]),
    Matrix([
        [0, -1, 0],
        [0,  0, 1],
        [0,  0, 0]
    ]),
    Matrix([
        [0, -1, 0],
        [0,  0, 1],
        [0,  0, 0]
    ])
]

# T23 >= T24, restriction
T23_to_T24 = [
    Matrix([
        [-1, 0, 0],
        [ 0, 0, 0],
        [ 0, 0, 0]
    ]),
    Matrix([
        [0, -1, 0],
        [0,  0, 0],
        [0,  0, 0]
    ]),
    Matrix([
        [0, 1, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
]

# T24 >= T25, restriction
T24_to_T25 = [
    Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]),
    Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]),
    Matrix([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
]

# Tdet + e(0,0,0) >= T10, d=(1,1,0)
D_111_to_10 = [
    Matrix([
        [1/x, 0, -1/x],
        [  0, 1,    0],
        [  0, 0,   -x]
    ]),
    Matrix([
        [1/x,  0,  0],
        [  0, -1, -1],
        [  0,  0,  x]
    ]),
    Matrix([
        [ -1, -1,  0],
        [  0,  0, -x],
        [x^2,  0,  0]
    ])
]

# Tdet + W >= T6, d=(2,0,0)
D_W_to_6 = [
    Matrix([
        [1/x,      -1/3/x,       0],
        [  0, -1/3*x^(-2), -x^(-2)],
        [  0,           1,       0]
    ]),
    Matrix([
        [-x, -1/3*x,  0],
        [ 0,   -1/3, -1],
        [ 0,   -x^2,  0]
    ]),
    Matrix([
        [-1,    0,  0],
        [ 0,    x,  0],
        [ 0, -1/3, -1]
    ])
]

# Tdet >= T17, restriction
D_to_17 = [
    Matrix([
        [ 0, 0, -1],
        [-1, 0,  0],
        [ 0, 0,  0]
    ]),
    Matrix([
        [0, 1,  0],
        [1, 0,  0],
        [0, 0, -1]
    ]),
    Matrix([
        [0, -1,  0],
        [1,  0,  0],
        [0,  0, -1]
    ])
]

# Verification
print("1->2                ", vector(apply(T1_to_T2, T1).flatten()).subs({x: 0}) == vector(T2.flatten()))
print("2->3                ", vector(apply(T2_to_T3, T2).flatten()).subs({x: 0}) == vector(T3.flatten()))
print("2->4                ", vector(apply(T2_to_T4, T2).flatten()).subs({x: 0}) == vector(pi(T4,1).flatten()))
print("3->5                ", vector(apply(T3_to_T5, T3).flatten()).subs({x: 0}) == vector(pi(T5,1).flatten()))
print("3->6                ", vector(apply(T3_to_T6, T3).flatten()).subs({x: 0}) == vector(T6.flatten()))
print("4->5                ", vector(apply(T4_to_T5, T4).flatten()).subs({x: 0}) == vector(T5.flatten()))
print("4->6                ", vector(apply(T4_to_T6, T4).flatten()).subs({x: 0}) == vector(T6.flatten()))
print("5->7                ", vector(apply(T5_to_T7, T5).flatten()).subs({x: 0}) == vector(T7.flatten()))
print("5->8                ", vector(apply(T5_to_T8, T5).flatten()).subs({x: 0}) == vector(T8.flatten()))
print("5->9                ", np.all(apply(T5_to_T9, T5) == pi(T9,2)))
print("6->7                ", vector(apply(T6_to_T7, T6).flatten()).subs({x: 0}) == vector(T7.flatten()))
print("7->10               ", vector(apply(T7_to_T10, T7).flatten()).subs({x: 0}) == vector(T10.flatten()))
print("7->11               ", vector(apply(T7_to_T11, T7).flatten()).subs({x: 0}) == vector(T11.flatten()))
print("7->12               ", np.all(apply(T7_to_T12, T7) == pi(T12,1)))
print("8->10               ", vector(apply(T8_to_T10, T8).flatten()).subs({x: 0}) == vector(T10.flatten()))
print("8->12               ", np.all(apply(T8_to_T12, T8) == T12))
print("9->12               ", vector(apply(T9_to_T12, T9).flatten()).subs({x: 0}) == vector(T12.flatten()))
print("10->13              ", np.all(apply(T10_to_T13, T10) == T13))
print("10->14              ", vector(apply(T10_to_T14, T10).flatten()).subs({x: 0}) == vector(T14.flatten()))
print("11->14              ", vector(apply(T11_to_T14, T11).flatten()).subs({x: 0}) == vector(pi(T14,1).flatten()))
print("11->15              ", np.all(apply(T11_to_T15, T11) == pi(T15,1)))
print("12->13              ", vector(apply(T12_to_T13, T12).flatten()).subs({x: 0}) == vector(T13.flatten()))
print("12->15              ", vector(apply(T12_to_T15, T12).flatten()).subs({x: 0}) == vector(T15.flatten()))
print("13->16              ", vector(apply(T13_to_T16, T13).flatten()).subs({x: 0}) == vector(T16.flatten()))
print("13->17              ", vector(apply(T13_to_T17, T13).flatten()).subs({x: 0}) == vector(T17.flatten()))
print("13->18              ", np.all(apply(T13_to_T18, T13) == T18))
print("14->16              ", np.all(apply(T14_to_T16, T14) == T16))
print("14->17              ", np.all(apply(T14_to_T17, T14) == pi(T17,2)))
print("15->16              ", vector(apply(T15_to_T16, T15).flatten()).subs({x: 0}) == vector(T16.flatten()))
print("16->19              ", np.all(apply(T16_to_T19, T16) == T19))
print("16->22              ", np.all(apply(T16_to_T22, T16) == T22))
print("17->19              ", np.all(apply(T17_to_T19, T17) == T19))
print("18->19              ", vector(apply(T18_to_T19, T18).flatten()).subs({x: 0}) == vector(T19.flatten()))
print("19->20              ", np.all(apply(T19_to_T20, T19) == T20))
print("20->21              ", vector(apply(T20_to_T21, T20).flatten()).subs({x: 0}) == vector(T21.flatten()))
print("21->23              ", np.all(apply(T21_to_T23, T21) == T23))
print("22->23              ", np.all(apply(T22_to_T23, T22) == T23))
print("23->24              ", np.all(apply(T23_to_T24, T23) == T24))
print("24->25              ", np.all(apply(T24_to_T25, T24) == T25))
print("D+111->10           ", vector(apply(D_111_to_10, Tdet + e(0,0,0)).flatten()).subs({x: 0}) == vector(T10.flatten()))
print("D+W->6              ", vector(apply(D_W_to_6, Tdet + W).flatten()).subs({x: 0}) == vector(T6.flatten()))
print("D->17               ", np.all(apply(D_to_17, Tdet) == T17))
