import numpy as np
import trimesh
import torch
import FastGeodis
import time


def load_obj(path):
    vertices, faces = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(list(map(float, line.split()[1:4])))
            elif line.startswith('f '):
                faces.append([int(x.split('/')[0]) - 1 for x in line.split()[1:4]])
    return np.array(vertices), np.array(faces)


# --- 1. Load meshes ---
print("Loading meshes...")
cage_verts, cage_faces = load_obj("data/armadilloman_cages_triangulated.obj")
source_verts, _ = load_obj("data/armadilloman.obj")
print(f"Cage mesh:     {len(cage_verts)} vertices, {len(cage_faces)} faces")
print(f"Source points: {len(source_verts)} vertices")

# --- 2. Voxelize the cage mesh ---
cage_mesh = trimesh.Trimesh(vertices=cage_verts, faces=cage_faces, process=False)
target_voxels = 150
pitch = cage_mesh.extents.max() / target_voxels
print(f"Using pitch: {pitch:.4f} (extents: {cage_mesh.extents})")

voxel_grid = cage_mesh.voxelized(pitch=pitch).fill()
volume = voxel_grid.matrix.astype(np.float32)
print(f"Voxel grid shape: {volume.shape}")

# --- 3. Build cost tensor (1.0 inside, 1e10 outside) ---
cost_tensor = torch.from_numpy(
    np.where(volume > 0, 1.0, 1e10).astype(np.float32)
).unsqueeze(0).unsqueeze(0)

# --- 4. Map vertices to voxel indices ---
cage_idx = voxel_grid.points_to_indices(cage_verts)      # (N_cage, 3)
source_idx = voxel_grid.points_to_indices(source_verts)  # (N_source, 3)

cage_inside = (volume[cage_idx[:, 0], cage_idx[:, 1], cage_idx[:, 2]] > 0).sum()
source_inside = (volume[source_idx[:, 0], source_idx[:, 1], source_idx[:, 2]] > 0).sum()
print(f"Cage verts inside voxel grid:   {cage_inside}/{len(cage_verts)}")
print(f"Source verts inside voxel grid: {source_inside}/{len(source_verts)}")

# --- 5. Deduplicate seed voxels ---
# Multiple source vertices may land in the same voxel; run FastGeodis once per unique voxel.
unique_voxels, source_to_unique = np.unique(source_idx, axis=0, return_inverse=True)
N_unique = len(unique_voxels)
N_source = len(source_verts)
N_cage = len(cage_verts)
print(f"Unique seed voxels: {N_unique} (from {N_source} source vertices)")
print(f"Output matrix will be: [{N_source} x {N_cage}]")

# --- 6. Allocate result matrix and pre-build softmask ---
# dist_matrix[i, j] = interior geodesic distance from source_verts[i] to cage_verts[j]
# in world-space units (distances from FastGeodis are multiplied by 2 because lamb=0.5
# scales interior steps by 0.5; outside voxels get np.nan).
dist_matrix = np.full((N_source, N_cage), np.nan, dtype=np.float32)

# Reuse a single softmask tensor — 1 = non-seed, 0 = seed
softmask = torch.ones(1, 1, *volume.shape, dtype=torch.float32)

CHECKPOINT_EVERY = 500  # save partial results every N unique voxels

# --- 7. Main loop ---
print("\nStarting distance computation...")
t0 = time.time()

for ui, (si, sj, sk) in enumerate(unique_voxels):
    softmask[0, 0, si, sj, sk] = 0.0

    dist_vol = FastGeodis.generalised_geodesic3d(
        cost_tensor, softmask, [pitch, pitch, pitch], 1e10, 0.5, 4
    ).squeeze().numpy()

    softmask[0, 0, si, sj, sk] = 1.0  # restore for next iteration

    # Sample at all cage vertices; ×2 converts 0.5-scaled distances to world units
    row = dist_vol[cage_idx[:, 0], cage_idx[:, 1], cage_idx[:, 2]] * 2.0

    # Write to all source vertices that share this voxel
    dist_matrix[source_to_unique == ui] = row

    if (ui + 1) % 50 == 0 or ui == 0:
        elapsed = time.time() - t0
        per_iter = elapsed / (ui + 1)
        eta = per_iter * (N_unique - ui - 1)
        print(f"  [{ui+1:5d}/{N_unique}]  elapsed: {elapsed:7.1f}s  ETA: {eta:7.0f}s  "
              f"({per_iter*1000:.0f} ms/seed)")

    if (ui + 1) % CHECKPOINT_EVERY == 0:
        np.save("dist_matrix_partial.npy", dist_matrix)
        print(f"  Checkpoint saved at {ui+1} unique voxels.")

# --- 8. Save final result ---
elapsed_total = time.time() - t0
print(f"\nDone in {elapsed_total:.1f}s ({elapsed_total/N_unique*1000:.0f} ms/seed average)")
print(f"Distance matrix shape: {dist_matrix.shape}  (source x cage)")
print(f"  min: {np.nanmin(dist_matrix):.4f}  max: {np.nanmax(dist_matrix):.4f}  "
      f"nan%: {np.isnan(dist_matrix).mean()*100:.1f}%")

np.save("dist_matrix.npy", dist_matrix)
print("Saved to dist_matrix.npy")
