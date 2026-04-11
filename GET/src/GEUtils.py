import numpy as np
import torch


class RegularToRegular:
    def __init__(self, N):
        self.N = N
        self.A = self.get_dft_matrix_A()

    def regular_to_regular_basis(self):
        """
        Returns the basis of linear maps W_i that satisfy the equivariance condition:
        rho @ W_i = W_i @ rho
        where rho is the regular representation. In this case, since regular to regular, a basis is given by circulant matrices
        """
        basis = []
        for i in range(self.N):
            W_i = np.zeros((self.N, self.N))
            for j in range(self.N):
                W_i[j, (j + i) % self.N] = 1.0
            basis.append(W_i)

        return [torch.tensor(v, dtype=torch.float32) for v in basis]

    def get_dft_matrix_A(self):
        A = torch.zeros((self.N, self.N))
        A[:, 0] = 1.0 / np.sqrt(self.N)

        for k in range(1, (self.N // 2) + 1):
            for j in range(self.N):
                angle = (2 * np.pi * j * k) / self.N
                A[j, 2 * k - 1] = np.sqrt(2.0 / self.N) * np.cos(angle)
                A[j, 2 * k] = np.sqrt(2.0 / self.N) * np.sin(angle)
        return A

    def extended_regular_representation(self, theta):
        """
        Builds the extended regular representation rho_tilde(theta)
        for a rotation by angle theta using the DFT-based change of basis A
        """
        # D_theta è una matrice a blocchi diagonali
        D_theta = torch.eye(self.N)
        # Il blocco (0,0) è 1 (già impostato con eye)

        for k in range(1, (self.N // 2) + 1):
            cos_kt = torch.cos(torch.tensor(k * theta))
            sin_kt = torch.sin(torch.tensor(k * theta))
            # Matrice di rotazione per la k-esima irrep
            R_k = torch.tensor([[cos_kt, -sin_kt], [sin_kt, cos_kt]])
            D_theta[2 * k - 1 : 2 * k + 1, 2 * k - 1 : 2 * k + 1] = R_k

        # rho_tilde(theta) = A @ D_theta @ A.T
        return self.A @ D_theta @ self.A.T


class LocalToRegular:
    def __init__(self, N):
        self.N = N
        self.rho_in = self.get_local_representation_rho_in()
        self.rho_out = self.get_regular_representation_rho_out()

    def local_to_regular_basis(self):
        """
        Computes the basis of linear maps W_i that satisfy the equivariance condition:
        rho_regular @ W_i = W_i @ rho_local
        """

        # Based on Eqn. (78), for n=0: (rho_out_inv \otimes rho_in) - I
        I_in = np.eye(3)
        I_out = np.eye(self.N)
        M = np.kron(I_out, self.rho_in.T) - np.kron(self.rho_out, I_in)

        # 3. Solve via SVD
        u, s, vh = np.linalg.svd(M)

        # The basis vectors are the rows of vh corresponding to zero singular values
        tol = 1e-10
        basis_vectors = vh[s < tol]

        # 4. Reshape to get W_i matrices (N x 3)
        return [
            torch.tensor(v.reshape(self.N, 3), dtype=torch.float32)
            for v in basis_vectors
        ]

    def get_local_representation_rho_in(self):
        # Rappresentazione locale: rotazione di 2pi/N intorno all'asse z
        theta = 2 * np.pi / self.N
        return np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ]
        )

    def get_regular_representation_rho_out(self):
        rho_out = np.zeros((self.N, self.N))
        for i in range(self.N):
            rho_out[(i + 1) % self.N, i] = 1
        return rho_out


if __name__ == "__main__":
    N = 9
    print("Regular to Regular basis:")
    r2r = RegularToRegular(N)
    W_r2r = r2r.regular_to_regular_basis()
    for i, W in enumerate(W_r2r):
        print(f"W_{i}:\n{W}\n")
