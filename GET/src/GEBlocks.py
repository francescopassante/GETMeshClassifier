import GEUtils
import numpy as np
import torch
import torch.nn as nn


class LocalToRegularLinearBlock(nn.Module):
    """
    A linear layer that implements equivariance from the rho_local representation to the regular representation for the cyclic group C_N.
    """

    def __init__(self, N, num_fields):
        """
        Args:
            N: Dimension of the regular representation (C_N).
            num_fields: Number of regular output fields.
        """
        super().__init__()
        self.N = N
        utils = GEUtils.LocalToRegular(N)
        W_basis = utils.local_to_regular_basis()
        self.register_buffer(
            "basis", torch.stack(W_basis)
        )  # This registers the basis as a non-learnable buffer
        self.num_basis = self.basis.shape[0]  # Number of basis matrices
        self.num_fields = num_fields

        # Learnable coefficients for each basis matrix for each output field
        # Initializing with small random values
        self.weights = nn.Parameter(torch.randn(num_fields, self.num_basis) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Input features (rho_local) of shape (Batch, Num_Points, 3)
        Returns:
            Output feature fields of shape (Batch, Num_Points, num_fields * N)
        """
        # 1. Compute the kernel for each field: W = sum(a_i * W_basis_i)
        # Resulting shape: (num_fields, N, 3)
        combined_kernels = torch.einsum("fk,knm->fnm", self.weights, self.basis)

        # 2. Reshape kernels to a single large weight matrix for efficient computation
        # Shape: (num_fields, N, 3) -> (num_fields * N, 3)
        W_final = combined_kernels.view(self.num_fields * self.N, 3)

        # 3. Apply the linear transformation to the input features
        # (B, P, 3) @ (3, num_fields*N) -> (B, P, num_fields*N)
        out = torch.matmul(x, W_final.t())

        return out


class SelfAttentionBlock(nn.Module):
    def __init__(self, N):
        super().__init__()
        self.N = N
        # self.n_heads = n_heads
        # self.d_k = out_channels // n_heads

        # Equivariant basis for Query and Key linear maps
        basis = GEUtils.RegularToRegular(N).regular_to_regular_basis()
        self.register_buffer("reg_to_reg_basis", torch.stack(basis))

        self.query_coeffs = nn.Parameter(torch.randn(len(basis)) * 0.02)
        self.key_coeffs = nn.Parameter(torch.randn(len(basis)) * 0.02)

        # The value matrix is given by a second order Taylor expansion in the relative position u.
        # The Taylor coefficients (matrices) must satisfy the equivariance condition in Eqn. (78) of the paper.
        # One finds that the equivariance is satisfied order by order, so there are separate bases for the zero, first, and second order terms.
        # So we allow a linear combination of each basis inside a given order.
        # We learn a linear combination of these basis matrices as the value function W_V(u).

        value_basis = GEUtils.RegularToRegular(N).get_taylor_basis()
        self.register_buffer("value_basis_zero_order", value_basis[0])
        self.register_buffer("value_basis_first_order", value_basis[1])
        self.register_buffer("value_basis_second_order", value_basis[2])

        self.value_matrix_zero_order_params = nn.Parameter(
            torch.randn(self.value_basis_zero_order.shape[0]) * 0.02
        )
        self.value_matrix_first_order_params = nn.Parameter(
            torch.randn(self.value_basis_first_order.shape[0]) * 0.02
        )
        self.value_matrix_second_order_params = nn.Parameter(
            torch.randn(self.value_basis_second_order.shape[0]) * 0.02
        )

        # Value Kernel: W_V(u) is represented by a set of precomputed bases
        # that satisfy gauge equivariance constraints [cite: 309, 1408]
        # Here we assume you have your equivariant 'kernel_bases' precomputed
        self.register_buffer(
            "kernel_bases", torch.randn(6, N, N)
        )  # Placeholder for 6 Taylor bases
        self.kernel_coeffs = nn.Parameter(torch.randn(n_heads, 6))

    def forward(self, x, neighbors, parallel_transport_matrices, rel_pos_u):
        """
        Args:
            x: [N_v, in_channels * N] - Center features
            neighbors: [N_v, Max_Neighbors] - Indices of neighbors
            parallel_transport_matrices: [N_v, Max_Neighbors, N, N] - rho_tilde(theta)
            rel_pos_u: [N_v, Max_Neighbors, 2] - Logarithmic map coordinates u_q
        """
        N_v, _ = x.shape

        # 1. Parallel Transport neighbors to center frame
        # x_neighbors shape: [N_v, Max_N, in_channels * N]
        x_neigh = x[neighbors]
        # Apply rho_tilde(theta) to each channel
        x_neigh = x_neigh.view(N_v, -1, -1, self.N)
        f_prime_q = torch.einsum(
            "vnoj,vnpj->vnop", parallel_transport_matrices, x_neigh
        )
        f_prime_q = f_prime_q.reshape(N_v, -1, x.shape[-1])

        # 2. Compute Attention Scores
        Q = self.W_Q(x).view(N_v, self.n_heads, self.d_k)
        K = self.W_K(f_prime_q).view(N_v, -1, self.n_heads, self.d_k)

        # Energy S(K, Q) - typically dot product scaled by sqrt(d_k)
        attn_scores = torch.einsum("vhd,vnhd->vnh", Q, K) / (self.d_k**0.5)
        attn_weights = F.softmax(attn_scores, dim=1)  # [N_v, Max_N, n_heads]

        # 3. Compute Values using Equivariant Kernel W_V(u)
        # W_V(u) = W0 + W1*u1 + W2*u2 ... (Taylor Expansion) [cite: 309, 1158]
        # For simplicity, we implement W_V(u) as a linear combination of equivariant bases
        W_V_u = torch.einsum(
            "h b, b i j -> h i j", self.kernel_coeffs, self.kernel_bases
        )

        # Apply value function to transported features
        # V = W_V(u) * f_prime_q
        f_prime_q_split = f_prime_q.view(N_v, -1, -1, self.N)
        values = torch.einsum("hij, vncj -> vnhci", W_V_u, f_prime_q_split)
        values = values.reshape(N_v, -1, self.n_heads, -1)

        # 4. Aggregation
        out = torch.einsum("vnh, vnhd -> vhd", attn_weights, values)
        return out.reshape(N_v, -1)


if __name__ == "__main__":
    # I want to check that if i rotate an input (x,y,z) then apply the layer i get a permutation of the output fields:
    def check_equivariance():
        def rotate_input(x, theta):
            # Rotate around the z-axis by angle theta
            cos_theta = torch.cos(theta)
            sin_theta = torch.sin(theta)
            rotation_matrix = torch.tensor(
                [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
                dtype=torch.float32,
            )
            return x @ rotation_matrix.t()

        equivariant_layer = LocalToRegularLinearBlock(N=9, num_fields=12)

        # Rotate the input by 2pi/9 (the angle corresponding to the cyclic group C9) and check the output
        theta = torch.tensor(2 * np.pi / 9, dtype=torch.float32)

        input = torch.randn(1, 1, 3)  # Original input
        rotated_input = rotate_input(input, theta)

        output = equivariant_layer(input).view(1, 1, 12, 9)
        rotated_output = equivariant_layer(rotated_input).view(1, 1, 12, 9)

        print(output[0][0][3])  # Should be (1, 1, 12, 9)
        print(rotated_output[0][0][3])  # Should be (1, 1, 12, 9)

    check_equivariance()
