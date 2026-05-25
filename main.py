import numpy as np
import trimesh
import torch
import FastGeodis


def load_obj(path):
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(vertices), np.array(faces)


# --- 1. Load mesh ---
verts, faces = load_obj("data/armadilloman.obj")
mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
print("Vertices:", len(verts))
print("Faces:", len(faces))
print("Bounds:", mesh.bounds)
print("Is watertight:", mesh.is_watertight)

# --- 2. Voxelize ---
target_voxels = 150
pitch = mesh.extents.max() / target_voxels
print(f"Using pitch: {pitch:.4f} (extents: {mesh.extents})")

voxel_grid = mesh.voxelized(pitch=pitch).fill()
volume = voxel_grid.matrix.astype(np.float32)  # (X, Y, Z), True inside
print("Voxel grid shape:", volume.shape)

# --- 3. Build cost tensor (inside=1.0, outside=1e10) ---
cost = np.where(volume > 0, 1.0, 1e10).astype(np.float32)
cost_tensor = torch.from_numpy(cost).unsqueeze(0).unsqueeze(0)  # (1, 1, X, Y, Z)

# --- 4. Place seed at your interior point ---
interior_point_world = mesh.centroid

seed = np.ones_like(volume, dtype=np.float32)  # FastGeodis: non-seed = 1, seed = 0
idx = voxel_grid.points_to_indices(interior_point_world.reshape(1, 3))[0]
# trimesh returns (x, y, z) but double-check axis order:
ix, iy, iz = idx[0], idx[1], idx[2]
print(f"Seed voxel index: ({ix}, {iy}, {iz}), inside={volume[ix, iy, iz] > 0}")
seed[ix, iy, iz] = 0.0              # <-- 0.0 marks the seed, NOT 1.0
seed_tensor = torch.from_numpy(seed).unsqueeze(0).unsqueeze(0)

# --- 5. Run FastGeodis ---
geodesic_dist = FastGeodis.generalised_geodesic3d(
    cost_tensor,
    seed_tensor,
    [pitch, pitch, pitch],
    1e10,   # v  (large = impassable outside)
    0.5,    # lambda: mix geodesic+euclidean so interior steps cost 0.5*pitch (not 0)
    4,      # n_iters
)

dist_volume = geodesic_dist.squeeze().numpy()

# --- 6. Query distance at a target vertex ---
target_vertex_world = verts[0]

idx_t = voxel_grid.points_to_indices(target_vertex_world.reshape(1, 3))[0]
tx, ty, tz = idx_t[0], idx_t[1], idx_t[2]
print(f"Target voxel index: ({tx}, {ty}, {tz}), inside={volume[tx, ty, tz] > 0}")

distance = dist_volume[tx, ty, tz]
print(f"Interior geodesic distance: {distance:.4f}  (scaled by 0.5; multiply by 2 for true path length)")