import GEUtils
import numpy as np
import torch
import torch.nn as nn


class LinearEquivariant(nn.Module):
    """
    A linear layer that implements the equivariant transformation using a learned combination of SVD-solved bases.
    It maps input features in the rho_local representation (3D coordinates) to output features in the regular representation (12 fields, each of dimension 9).
    """

    def __init__(self, W_basis, num_fields=12):
        """
        Args:
            W_basis: List of torch.Tensors, each of shape (9, 3).
                     These are the SVD-solved equivariant bases.
            num_fields: Number of regular output fields (12 for SHREC).
        """
        super().__init__()
        self.register_buffer(
            "basis", torch.stack(W_basis)
        )  # This registers the basis as a non-learnable buffer
        self.num_basis = self.basis.shape[0]
        self.num_fields = num_fields

        # Learnable coefficients for each basis matrix for each output field
        # Initializing with small random values
        self.weights = nn.Parameter(torch.randn(num_fields, self.num_basis) * 0.02)

    def forward(self, x):
        """
        Args:
            x: Input features (rho_local) of shape (Batch, Num_Points, 3)
        Returns:
            Output feature fields of shape (Batch, Num_Points, 108)
        """
        # 1. Compute the kernel for each field: W = sum(a_i * W_basis_i)
        # Resulting shape: (num_fields, 9, 3)
        combined_kernels = torch.einsum("fk,knm->fnm", self.weights, self.basis)

        # 2. Reshape kernels to a single large weight matrix for efficient computation
        # Shape: (num_fields * 9, 3) -> (108, 3)
        W_final = combined_kernels.view(self.num_fields * 9, 3)

        # 3. Apply the linear transformation to the input features
        # (B, P, 3) @ (3, 108) -> (B, P, 108)
        out = torch.matmul(x, W_final.t())

        return out


if __name__ == "__main__":
    # I want to check that if i rotate an input (x,y,z) then apply the layer i get a permutation of the output fields:

    def rotate_input(x, theta):
        # Rotate around the z-axis by angle theta
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        rotation_matrix = torch.tensor(
            [[cos_theta, -sin_theta, 0], [sin_theta, cos_theta, 0], [0, 0, 1]],
            dtype=torch.float32,
        )
        return x @ rotation_matrix.t()

    utils = GEUtils.Linear(N=9)
    W_basis = utils.compute_W_basis()  # This should give us a list of numpy arrays
    W_basis_torch = [torch.tensor(W, dtype=torch.float32) for W in W_basis]
    equivariant_layer = LinearEquivariant(W_basis=W_basis_torch, num_fields=12)

    # Rotate the input by 2pi/9 (the angle corresponding to the cyclic group C9) and check the output
    theta = torch.tensor(2 * 8 * np.pi / 9, dtype=torch.float32)

    input = torch.randn(1, 1, 3)  # Original input
    rotated_input = rotate_input(input, theta)

    output = equivariant_layer(input).view(1, 1, 12, 9)
    rotated_output = equivariant_layer(rotated_input).view(1, 1, 12, 9)

    print(output[0][0][3])  # Should be (1, 1, 12, 9)
    print(rotated_output[0][0][3])  # Should be (1, 1, 12, 9)
