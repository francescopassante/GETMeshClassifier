import GEUtils
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


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
    def __init__(self, N, in_channels):
        super().__init__()
        self.N = N
        # self.n_heads = n_heads
        # self.d_k = out_channels // n_heads

        self.in_channels = in_channels

        # Equivariant basis for Query and Key linear maps
        basis = GEUtils.RegularToRegular(N).regular_to_regular_basis()
        self.register_buffer("reg_to_reg_basis", torch.stack(basis))

        # Query and Key coefficients are [in_channels, len_basis] because we use a linear comb of the basis for each channel, then sum
        self.query_coeffs = nn.Parameter(torch.randn(in_channels, len(basis)) * 0.02)
        self.key_coeffs = nn.Parameter(torch.randn(in_channels, len(basis)) * 0.02)

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
            torch.randn(in_channels, self.value_basis_zero_order.shape[0]) * 0.02
        )
        self.value_matrix_first_order_params = nn.Parameter(
            torch.randn(in_channels, self.value_basis_first_order.shape[0]) * 0.02
        )
        self.value_matrix_second_order_params = nn.Parameter(
            torch.randn(in_channels, self.value_basis_second_order.shape[0]) * 0.02
        )

    def W_K(self, x):
        # key_coeffs are [in_channels, len_basis]
        # reg_to_reg_basis are [len_basis, N, N] (len_basis = N tra l'altro)
        # W_K is a NxN matrix for each channel -> [in_channels, N, N]

        W_K = torch.einsum("cb, bij -> cij", self.key_coeffs, self.reg_to_reg_basis)
        x = x.view(x.shape[0], self.in_channels, self.N)  # x = [N_v, in_channels, N]
        return torch.einsum("cij, vcj -> vi", W_K, x)

    def W_Q(self, fprime):
        # fprime is [N_v, MAX_NEIGH, in_channels, N]
        # W_Q is [in_channels, N, N]
        W_Q = torch.einsum("cb, bij -> cij", self.query_coeffs, self.reg_to_reg_basis)
        return torch.einsum("cij, vncj -> vni", W_Q, fprime)

    def forward(self, x, neighbors, mask, parallel_transport_matrices, rel_pos_u):
        """
        Args:
            x: [N_v, in_channels * N] - Center features
            neighbors: [N_v, Max_Neighbors] - Indices of neighbors
            mask: [N_v, Max_Neighbors] - Binary mask for valid neighbors
            parallel_transport_matrices: [N_v, Max_Neighbors, N, N] - rho_tilde(theta)
            rel_pos_u: [N_v, Max_Neighbors, 2] - Logarithmic map coordinates u_q
        """
        N_v, _ = x.shape

        # 1. Parallel Transport neighbors to center frame
        # x_neighbors shape: [N_v, Max_N, in_channels * N]
        x_neigh = x[neighbors]

        x_neigh = x_neigh.view(N_v, -1, self.in_channels, self.N)

        # Zero-out "fake neighbors" so that x_neigh[v][n] is zero if n > actual number of neighbors for vertex v
        mask_expanded = mask.unsqueeze(-1).unsqueeze(-1)  # [N_v, Max_N, 1, 1]
        x_neigh = x_neigh * mask_expanded

        # Apply rho_tilde(theta) to each channel
        f_prime_q = torch.einsum(
            "vnij,vncj->vnci", parallel_transport_matrices, x_neigh
        )
        print("f'_q shape: ", f_prime_q.shape)

        # 2. Compute Attention Scores
        # print("x shape: ", x.shape)
        K = self.W_K(x)  # .view(N_v, -1, self.n_heads, self.d_k)
        print("K shape: ", K.shape)
        Q = self.W_Q(f_prime_q)
        print("Q shape: ", Q.shape)

        score = (
            torch.relu(Q + K.unsqueeze(1)).mean(dim=-1).masked_fill(~mask, 0)
        )  # [N_v, Max_Neigh, in_channels]

        print("Score shape: ", score.shape)
        score_denominator = score.sum(dim=-1).clamp(min=1e-8)

        attention = score / score_denominator.unsqueeze(-1)
        print("Attention shape: ", attention.shape)

        # 3. Compute Values using Equivariant Kernel W_V(u)
        # W_V(u) = W0 + W1*u1 + W2*u2 ... (Taylor Expansion)

        print("rel_pos_u shape: ", rel_pos_u.shape)
        u_0 = rel_pos_u[..., 0]
        u_1 = rel_pos_u[..., 1]
        u_0_squared = u_0**2
        u_1_squared = u_1**2
        u_0_u_1 = (
            2 * u_0 * u_1
        )  # This 2 factor i think is fundamental, goes back to the SVD solution and form of F for the second order

        zero_order = torch.einsum(
            "cb,boij->coij",
            self.value_matrix_zero_order_params,
            self.value_basis_zero_order,
        ).squeeze(1)  # [N, N]

        first_order = torch.einsum(
            "cb,boij,vno->vncij",
            self.value_matrix_first_order_params,
            self.value_basis_first_order,
            rel_pos_u,
        )

        second_order = torch.einsum(
            "cb,boij,vno->vncij",
            self.value_matrix_second_order_params,
            self.value_basis_second_order,
            torch.stack([u_0_squared, u_0_u_1, u_1_squared], dim=-1),
        )

        value_kernel = (
            zero_order.unsqueeze(0).unsqueeze(0) + first_order + second_order
        )  # [in_channels, N_v, Max_Neigh, N, N]

        # Apply value function to transported features
        # V = W_V(u) * f_prime_q
        f_prime_q = f_prime_q.view(N_v, -1, self.in_channels, self.N)

        values = torch.einsum("vncij,vncj->vnci", value_kernel, f_prime_q)

        # 4. Aggregation
        out = torch.einsum("vn,vnci->vci", attention, values)  # [N_v, in_channels, N]
        return out


class GroupPooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        return x.max(dim=-1)[0]  # [N_v, in_channels]


class GlobalAveragePooling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def forward(self, x):
        return x.mean(dim=0)  # [in_channels]


if __name__ == "__main__":
    # I want to check that if i rotate an input (x,y,z) then apply the layer i get a permutation of the output fields:
    def check_equivariance_l2r():
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

    def check_equivariance_sa(N, channels):
        # load the data
        path = "../data/processed/T3.pt"
        data = torch.load(path, map_location="cpu")
        x = data["features"]  # (N_v, 3)
        neighbors = data["neighbors"]  # (N_v, Max_Neighbors)
        parallel_transport_angles = data["g_qp"]  # (N_v, Max_Neighbors, N, N)
        rel_pos_u = data["u_q"]  # (N_v, Max_Neighbors, 2)
        mask = data["mask"]  # (N_v, Max_Neighbors)

        # (N_v, in_channels * N)
        l2rBlock = LocalToRegularLinearBlock(N, num_fields=channels)

        r2r = GEUtils.RegularToRegular(N)
        parallel_transport_matrices = r2r.extended_regular_representation(
            parallel_transport_angles
        )

        print("partranspmatr.shape", parallel_transport_matrices.shape)
        theta = torch.tensor(2 * np.pi / N)
        rot_mat_3d = torch.tensor(
            [
                [torch.cos(theta), -torch.sin(theta), 0],
                [torch.sin(theta), torch.cos(theta), 0],
                [0, 0, 1],
            ]
        )

        rot_x = torch.einsum("ij,vj->vi", rot_mat_3d, x)

        input = l2rBlock(x)
        rot_input = l2rBlock(rot_x)

        # A rotation of the reference frame also requires a rotation of the relative positions!
        rot_rel_pos_u = torch.einsum("ij,vnj->vni", rot_mat_3d[:2, :2], rel_pos_u)

        sa = SelfAttentionBlock(N, in_channels=channels)

        output = sa(input, neighbors, mask, parallel_transport_matrices, rel_pos_u)
        rot_output = sa(
            rot_input, neighbors, mask, parallel_transport_matrices, rot_rel_pos_u
        )

        print(output[0])
        print(rot_output[0])

    def show_pooling():
        group_pool = GroupPooling(in_channels=3)
        input = torch.tensor(
            [
                [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]],
                [[25, 26, 27, 28], [29, 30, 31, 32], [33, 34, 35, 36]],
            ],
            dtype=torch.float32,
        )
        print(out := group_pool(input))
        print(out.shape)

        global_pool = GlobalAveragePooling(in_channels=3)
        print(out := global_pool(out))
        print(out.shape)

    def check_gauge_invariance(data, angles, N, channels, verbose=True):
        # This tests wether the network (local2reg linear block, self attention block, group pool, global average pool)
        # is gauge invariant, performing a different rotation for each vertex

        x = data["features"]  # (N_v, 3)
        neighbors = data["neighbors"]  # (N_v, Max_Neighbors)
        parallel_transport_angles = data["g_qp"]  # (N_v, Max_Neighbors)

        rel_pos_u = data["u_q"]  # (N_v, Max_Neighbors, 2)
        mask = data["mask"]  # (N_v, Max_Neighbors)

        l2rBlock = LocalToRegularLinearBlock(N, num_fields=channels)

        r2r = GEUtils.RegularToRegular(N)
        parallel_transport_matrices = r2r.extended_regular_representation(
            parallel_transport_angles
        )

        sa = SelfAttentionBlock(N, in_channels=channels)

        # The parallel transport angles transform as: new_theta_nv = theta_nv + random_angle_v - random_angle_n
        new_parallel_transport_angles = (
            parallel_transport_angles + angles.unsqueeze(-1) - angles[neighbors]
        )

        rot_parallel_transport_matrices = r2r.extended_regular_representation(
            new_parallel_transport_angles
        )

        cos = torch.cos(angles)  # (N_v,)
        sin = torch.sin(angles)  # (N_v,)

        # fmt: off
        rot_mat_3d = torch.stack([
            torch.stack([cos, -sin, torch.zeros_like(cos)], dim=-1),
            torch.stack([sin,  cos, torch.zeros_like(cos)], dim=-1),
            torch.stack([torch.zeros_like(cos), torch.zeros_like(cos), torch.ones_like(cos)], dim=-1)
        ], dim=-2)
        # fmt: on

        x_rot = torch.einsum("vij,vj->vi", rot_mat_3d, x)
        rot_rel_pos_u = torch.einsum("vij,vnj->vni", rot_mat_3d[:, :2, :2], rel_pos_u)

        input = l2rBlock(x)
        rot_input = l2rBlock(x_rot)

        output = sa(input, neighbors, mask, parallel_transport_matrices, rel_pos_u)
        rot_output = sa(
            rot_input, neighbors, mask, rot_parallel_transport_matrices, rot_rel_pos_u
        )

        # Now let's apply group pooling and average pooling:
        group_poool = GroupPooling(in_channels=channels)
        global_pool = GlobalAveragePooling(in_channels=channels)

        out = global_pool(group_poool(output))
        rot_out = global_pool(group_poool(rot_output))

        if verbose:
            print(
                "gauge rotation at [0] in terms of 2pi/N: ",
                angles[0] / (2 * np.pi / N),
            )
            print("Original output of self attention block, [0]: \n", output[0])
            print(
                "New output of self attention block with changed gauge, [0]: \n",
                rot_output[0],
            )
            print("out: ", out)
            print("rot_out: ", rot_out)

        return out, rot_out

    path = "../data/processed/T3.pt"
    data = torch.load(path, map_location="cpu")

    def mean_gauge_violation(data, N, channels, trials):
        N_v = data["features"].shape[0]
        gauge_violation = 0
        for i in tqdm(range(trials)):
            angles = torch.randn((N_v,), dtype=torch.float32) * 2 * np.pi / N
            rot, rot_out = check_gauge_invariance(
                data, angles, N, channels, verbose=False
            )
            gauge_violation += torch.norm(rot - rot_out) / (
                (torch.norm(rot) + torch.norm(rot_out)) / 2
            )
        return gauge_violation / trials

    # gauge_violations = []
    # for N in tqdm([i for i in range(1, 11) if i % 2 == 1]):
    #     gauge_violations.append(mean_gauge_violation(data, N, 4, 3))
    # print([g.item() for g in gauge_violations])

    import matplotlib.pyplot as plt

    # runs = torch.tensor(
    #     [
    #         [0.0404, 0.0035, 0.0021, 0.00270, 0.0009, 0.00152],
    #         [0.0339, 0.0073, 0.0009, 0.0012, 0.0014, 0.00117],
    #         [0.0494, 0.0184, 0.0031, 0.0005, 0.0008, 0.00110],
    #     ]
    # ).mean(axis=0)

    runs = [
        0.46849676966667175,
        0.007196597754955292,
        0.0037321485579013824,
        0.0010028083343058825,
        0.0009637857438065112,
    ]

    plt.scatter(
        [i for i in range(3, 11) if i % 2 == 1], runs[1:], c="r", marker="*", s=80
    )
    plt.xlabel("N")
    plt.ylabel(r"$\frac{||GET(x) - GET(gx)||}{(||GET(x)||+||GET(gx)||)/2}$")
    plt.grid()
    plt.title("Gauge violation as a function of N, 4 channels, 3 runs averaged")
    plt.show()

    # 0.0015286189736798406, 0.0011697500012814999, 0.0011058930540457368
