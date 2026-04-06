import numpy as np
import potpourri3d as pp3d
import torch
import trimesh


class MeshPreprocessor:
    def __init__(self, path, subsample=0.1):
        self.mesh = self.preprocess_mesh(path, subsample)

    def preprocess_mesh(self, mesh_path, subsample):
        # Load the mesh
        mesh = trimesh.load(mesh_path)

        simplified_mesh = mesh.simplify_quadric_decimation(percent=1 - subsample)

        # we compute its surface area by summing up the areas of all faces, and then scale it into 1
        area = simplified_mesh.area
        if area > 0:
            simplified_mesh.apply_scale(1 / np.sqrt(area))

        return simplified_mesh

    def compute_geodesic_neighborhood(self, p_idx, distance):

        solver = pp3d.MeshHeatMethodDistanceSolver(self.mesh.vertices, self.mesh.faces)

        # 3. Compute distances from a source vertex p (index p_idx)
        distances = solver.compute_distance(p_idx)

        # 4. Select vertices within the 0.2 geodesic radius
        neighbor_indices = np.where(distances <= distance)[0]
        return neighbor_indices

    # Function to plot the neighbors of vertex 0:
    def plot_neighbors(self, p_idx, neighbor_indices):
        import matplotlib.pyplot as plt

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
        ax.set_title("Geodesic Neighborhood of Vertex 0")
        plt.show()


if __name__ == "__main__":
    # Example usage
    path = "data/SHREC11_test_database_new/T42.off"  # Replace with your mesh file path
    preprocessor = MeshPreprocessor(path, subsample=0.1)
    print("total vertices: ", len(preprocessor.mesh.vertices))
    neighbor_indices = preprocessor.compute_geodesic_neighborhood(
        p_idx=100, distance=0.2
    )
    print("neighbors: ", len(neighbor_indices))
    preprocessor.plot_neighbors(100, neighbor_indices)
    preprocessor.mesh.show()
