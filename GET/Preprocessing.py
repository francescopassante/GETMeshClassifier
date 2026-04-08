from multiprocessing import process
from os import path

import matplotlib.pyplot as plt
import numpy as np
import potpourri3d as pp3d
import torch
import trimesh
from tqdm import tqdm


class MeshPreprocessor:
    def __init__(self, mesh):
        self.mesh = mesh

    @classmethod
    def from_file(cls, mesh_path, subsample):
        mesh = cls.preprocess_mesh(cls, mesh_path, subsample)
        return cls(mesh)

    def __str__(self):
        return f"MeshPreprocessor(mesh with {len(self.mesh.vertices)} vertices and {len(self.mesh.faces)} faces)"

    def preprocess_mesh(self, mesh_path, subsample):
        # Load the mesh
        mesh = trimesh.load(mesh_path)

        # Subsample the mesh using quadric decimation to reduce the number of vertices while preserving the overall shape
        simplified_mesh = mesh.simplify_quadric_decimation(percent=1 - subsample)

        #  Normalize surface area to 1
        area = simplified_mesh.area
        if area > 0:
            simplified_mesh.apply_scale(1 / np.sqrt(area))

        return simplified_mesh

    def compute_geodesic_neighborhood(self, p_idx, radius):

        solver = pp3d.MeshHeatMethodDistanceSolver(self.mesh.vertices, self.mesh.faces)

        # 3. Compute distances from a source vertex p (index p_idx)
        distances = solver.compute_distance(p_idx)

        # 4. Select vertices within the 0.2 geodesic radius
        neighbor_indices = np.where(distances <= radius)[0]
        return neighbor_indices

    def compute_log_and_ptransport(self, radius=0.2, max_neighbors=200):
        """
        Efficiently precomputes logarithmic maps and transport angles for
        all neighborhoods in a single pass using the Vector Heat Method.

        This version avoids recomputing the same transported vector field
        multiple times: we first build the neighborhood lists for every
        center vertex, and record which centers need each source vertex q.
        Then we compute the transport field for each source q only once and
        fill the corresponding angles for every center that had q as a
        neighbor.
        """
        vertices = self.mesh.vertices
        faces = self.mesh.faces
        num_vertices = len(vertices)

        # 1. Initialize the Vector Heat Solvers
        # This pre-factors the Laplacian and Poisson matrices once
        dist_solver = pp3d.MeshHeatMethodDistanceSolver(vertices, faces)
        vector_solver = pp3d.MeshVectorHeatSolver(vertices, faces)

        # Prepare data structures
        neighbor_data = [None] * num_vertices
        # For each source vertex q, keep a list of center indices p that include q as a neighbor
        centers_per_q = [[] for _ in range(num_vertices)]
        # For each center p, map neighbor q -> position index inside that center's neighbor array
        positions_per_center = [dict() for _ in range(num_vertices)]

        # First pass: build neighborhoods and placeholders for g_qp
        for i in range(num_vertices):
            # Identify neighbors within geodesic radius
            dists = dist_solver.compute_distance(i)
            neighbor_indices = np.where(dists <= radius)[0]
            # Remove the center vertex from its own neighborhood
            neighbor_indices = neighbor_indices[neighbor_indices != i]

            # If there are more neighbors than max_neighbors, we keep the closest ones.
            if len(neighbor_indices) > max_neighbors:
                neighbor_indices = neighbor_indices[
                    np.argsort(dists[neighbor_indices])
                ][:max_neighbors]

            # Compute the Logarithmic Map u_q for the center i and keep only neighbors
            u_q = vector_solver.compute_log_map(i)[neighbor_indices]

            # Placeholder for angles; will be filled in the second pass
            g_qp = np.zeros(len(neighbor_indices), dtype=np.float32)

            # Store neighbor info
            neighbor_data[i] = {
                "q_indices": neighbor_indices.astype(np.int32),
                "u_q": u_q.astype(np.float32),
                "g_qp": g_qp,
            }

            # Record reverse mapping from neighbor q to center i and position
            for pos, q in enumerate(neighbor_indices):
                q_int = int(q)
                centers_per_q[q_int].append(i)
                positions_per_center[i][q_int] = pos

        # Second pass: compute transport fields once per source q and fill angles for all centers
        for q in range(num_vertices):
            centers = centers_per_q[q]
            if not centers:
                continue

            # Transport the canonical tangent vector (1,0) from q to all vertices
            transported_field = vector_solver.transport_tangent_vector(q, [1.0, 0.0])

            # Fill angles for every center p that had q as a neighbor
            for p in centers:
                pos = positions_per_center[p][q]
                v = transported_field[p]
                angle = np.arctan2(v[1], v[0])
                neighbor_data[p]["g_qp"][pos] = np.float32(angle)

        return neighbor_data

    # Function to plot the neighbors of vertex 0, debug purposes:
    def plot_neighbors(self, p_idx, distance):
        neighbor_indices = self.compute_geodesic_neighborhood(p_idx, distance)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            self.mesh.vertices[:, 0],
            self.mesh.vertices[:, 1],
            self.mesh.vertices[:, 2],
            color="lightgray",
            s=10,
        )
        ax.scatter(
            self.mesh.vertices[neighbor_indices, 0],
            self.mesh.vertices[neighbor_indices, 1],
            self.mesh.vertices[neighbor_indices, 2],
            color="red",
            s=10,
        )

        # Plot the source vertex in blue
        ax.scatter(
            self.mesh.vertices[p_idx, 0],
            self.mesh.vertices[p_idx, 1],
            self.mesh.vertices[p_idx, 2],
            color="blue",
            s=50,
        )
        ax.set_title(f"Geodesic Neighborhood of Vertex {p_idx}")
        plt.show()


if __name__ == "__main__":
    base = "data/SHREC11_test_database_new/"
    paths = [i for i in range(0, 600) if not path.exists(f"data/processed/T{i}.pt")]
    K = 200  # max neighbors (actual max is 318, i cap it, see neighborhood_sizes.png)

    for j, file_number in enumerate(tqdm(paths)):
        preprocessor = MeshPreprocessor.from_file(
            base + f"T{file_number}.off", subsample=0.1
        )
        faces_sorted = np.sort(preprocessor.mesh.faces, axis=1)
        has_duplicates = len(faces_sorted) != len(np.unique(faces_sorted, axis=0))
        try:
            neighbor_data = preprocessor.compute_log_and_ptransport(
                radius=0.2, max_neighbors=K
            )
        except Exception as e:
            print(f"Error processing file {file_number}: {e}")
            continue

        N = len(neighbor_data)  # number of vertices

        # Preallocate tensors
        neighbors = torch.full((N, K), -1, dtype=torch.long)  # neighbor indices
        u_q = torch.zeros((N, K, 2), dtype=torch.float32)  # 2D vectors
        g_qp = torch.zeros((N, K), dtype=torch.float32)  # cos/sin angles
        mask = torch.zeros((N, K), dtype=torch.bool)  # valid neighbors mask

        # Fill tensors
        for i, d in enumerate(neighbor_data):
            q_indices = d["q_indices"]
            n = min(len(q_indices), K)  # number of neighbors (capped at K)

            u = d["u_q"]
            g = d["g_qp"]

            # store neighbor indices
            neighbors[i, :n] = torch.from_numpy(q_indices)
            # store vectors
            u_q[i, :n] = torch.from_numpy(u)
            # store angles
            g_qp[i, :n] = torch.from_numpy(g)
            # mask
            mask[i, :n] = True

        # Save as a PyTorch file
        torch.save(
            {"neighbors": neighbors, "u_q": u_q, "g_qp": g_qp, "mask": mask},
            f"data/processed/T{file_number}.pt",
        )
        # Save the preprocessed mesh as well, for reference (optional)
        preprocessor.mesh.export(f"data/processed/T{file_number}.off")
