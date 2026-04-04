import numpy as np


class Linear:
    def __init__(self, N):
        self.N = N
        # 1. Define Representations
        theta = 2 * np.pi / N
        self.rho_in = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )
        # Regular representation (cyclic shift matrix)
        self.rho_out = np.zeros((N, N))
        for i in range(N):
            self.rho_out[(i + 1) % N, i] = 1

    def compute_W_basis(self):
        # Build the Constraint Matrix (M)
        # Based on Eqn. (78), for n=0: (rho_out_inv \otimes rho_in) - I
        # Note: For orthogonal representations, inverse is transpose [cite: 162]
        I_in = np.eye(3)
        I_out = np.eye(self.N)
        M = np.kron(I_out, self.rho_in.T) - np.kron(self.rho_out, I_in)

        # 3. Solve via SVD
        u, s, vh = np.linalg.svd(M)

        # The basis vectors are the rows of vh corresponding to zero singular values
        tol = 1e-10
        basis_vectors = vh[s < tol]

        # 4. Reshape to get W_i matrices (N x 3)
        return [v.reshape(self.N, 3) for v in basis_vectors]


if __name__ == "__main__":
    # Now i check that the W_i satisfy the equivariance condition:
    utils = Linear(N=9)
    W_basis = utils.compute_W_basis()
    rho_in = utils.rho_in
    rho_out = utils.rho_out

    def check_equivariance(W):
        lhs = rho_out @ W
        rhs = W @ rho_in
        if not np.allclose(lhs, rhs):
            return False
        return True

    # Check all basis vectors
    for i, W in enumerate(W_basis):
        if check_equivariance(W):
            print(f"W_{i} satisfies the equivariance condition.")
        else:
            print(f"W_{i} does NOT satisfy the equivariance condition.")

    # Now i want to check that applying a rotation to the input and then applying W i get a permutation of the output fields:
    def check_permutation(W, x):
        # Apply W to the original input
        original_output = W @ x

        # Rotate the input
        rotated_x = rho_in @ x

        # Apply W to the rotated input
        rotated_output = W @ rotated_x

        # Check if the outputs are permutations of each other
        return original_output, rotated_output

    print("\nChecking permutation property for W_0:")
    x = np.random.rand(3)  # Random input vector
    original_output, rotated_output = check_permutation(W_basis[0], x)
    print("Original output:", original_output)
    print("Rotated output:", rotated_output)
