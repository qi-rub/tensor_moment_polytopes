# The moment polytopes of tensors of shape 2x2x2 (up to permutations), and the
# Kronecker polytope (the maximal moment polytope).

kronecker_2x2x2 = [
        (1/2, 1/2,     1,   0,   1/2, 1/2,   0),
        (1/2, 1/2,   1/2, 1/2,     1,   0,   0),
        (  1,   0,   1/2, 1/2,   1/2, 1/2,   0),
        (  1,   0,     1,   0,     1,   0,   0),
        (1/2, 1/2,   1/2, 1/2,   1/2, 1/2,   0)
    ],

polytopes_2x2x2 = {
    # The moment polytope of U₂ = e₁⊗e₁⊗e₁ + e₂⊗e₂⊗e₂.
    'U2': kronecker_2x2x2,
    # The moment polytope of W = e₁⊗e₁⊗e₂ + e₁⊗e₂⊗e₁ + e₂⊗e₁⊗e₁.
    'W':
        [
            (  1,   0,     1,   0,     1,   0,  0),
            (1/2, 1/2,     1,   0,   1/2, 1/2,  0),
            (  1,   0,   1/2, 1/2,   1/2, 1/2,  0),
            (1/2, 1/2,   1/2, 1/2,     1,   0,  0)
        ],
    # The moment polytope of e₁⊗e₁⊗e₁ + e₁⊗e₂⊗e₂.
    'EPR':
        [
            (  1,   0,     1,   0,     1,   0,  0),
            (  1,   0,   1/2, 1/2,   1/2, 1/2,  0)
        ],
    # The moment polytope of U₁ = e₁⊗e₁⊗e₁.
    'U1':
        [
            (  1,   0,     1,   0,     1,   0,  0)
        ],
    'zero':
        [
        ]
    }



