import trimesh
import numpy as np

# Test trimesh functionality
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
test_points = np.array([[0.1, 0.1, 0.1], [2, 2, 2]])
inside = mesh.contains(test_points)

print(f'✅ Trimesh working!')
print(f'   - Mesh created: {mesh}')
print(f'   - Point containment: {inside}')
print(f'   - Watertight: {mesh.is_watertight}')
