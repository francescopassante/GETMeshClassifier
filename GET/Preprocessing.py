import potpourri3d as pp3d
import trimesh


class MeshPreprocessor:
    def __init__(self, path):
        self.mesh = self.preprocess_mesh(path)

    def preprocess_mesh(self, mesh_path):
        # Load the mesh
        mesh = trimesh.load(mesh_path)
        # we compute its surface area by summing up the areas of all faces, and then scale it into 1
        area = mesh.area
        if area > 0:
            mesh.apply_scale(1 / area)
        return mesh

    solver = pp3d.MeshHeatMethodDistanceSolver(self.mesh.vertices, self.mesh.faces)

    # 3. Compute distances from a source vertex p (index p_idx)
    p_idx = 0
    distances = solver.compute_distance(p_idx)

    # 4. Select vertices within the 0.2 geodesic radius
    neighborhood_mask = distances <= 0.2
    neighbor_indices = np.where(neighborhood_mask)[0]

    print(f"Vertex {p_idx} has {len(neighbor_indices)} neighbors within radius 0.2")


if __name__ == "__main__":
    # Example usage
    path = "data/T0.off"  # Replace with your mesh file path
    preprocessor = MeshPreprocessor(path).mesh.show()
