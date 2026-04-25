# Gauge Equivairant Transformer - Mesh Classifier
Gauge Equivariant Transformer for 3D mesh classification on the SHREC11 dataset.

---

This repository reproduces the SHREC11 classification experiment from the [Gauge Equivariant Transformer paper (He, Dong Wang, Tao, Lin, NeurIPS 2021)](https://proceedings.neurips.cc/paper/2021/file/e57c6b956a6521b28495f2886ca0977a-Paper.pdf). The dataset consists of 600 meshes divided into 30 shape classes (human, bird, alien, hand, ...) with 20 meshes per class.

---

## Pipeline overview

### 1. Preprocessing (`GEPreprocessing.py`)

Each raw SHREC11 mesh is processed by:

1. **Quadric decimation** (trimesh) — simplifies to ~20% of original vertices, yielding ~1000 points per mesh, normalized to unit surface area.
2. **Neighborhood computation** — for each vertex, geodesic neighbors within radius σ = 0.2 are collected (up to 50 per vertex).
3. **Log map and parallel transport** (potpourri3d / heat method) — for each vertex–neighbor pair, computes:
   - `u_q`: the 2D log-map coordinates of the neighbor in the center vertex's tangent frame.
   - `g_qp`: the parallel transport angle from the neighbor's frame to the center's frame.
4. **Local features** — vertex coordinates projected onto each vertex's tangent frame: `[<p, x>, <p, y>, <p, n>]` (a 3-vector in the local gauge).

Processed meshes are saved as `.pt` files containing `features`, `neighbors`, `u_q`, `g_qp`, and a validity `mask`.

---

### 2. Architecture (`GEBlocks.py`, `GET.py`)

The full model follows the architecture from the [supplementary material](https://openreview.net/attachment?id=fyL9HD-kImm&name=supplementary_material), with a few modifications (see below):

![Architecture of the Transformer](GET/images/architecture.png)

The forward pass is:

```
LocalToRegular → [GEResNetBlock × num_blocks] → GroupPooling → GlobalAveragePooling → Linear(channels, 30)
```

**Gauge equivariance** is enforced at every stage:

#### LocalToRegular (`GELocalToRegularLinearBlock`)
Maps each vertex's 3D local feature to a *regular field* of shape `(channels, N)`, where N is the order of the cyclic group C_N. The admissible equivariant kernels form a linear basis computed via SVD; learnable weights are a linear combination of these basis elements.

#### Gauge-equivariant Self-Attention (`GESelfAttentionBlock`)
For each vertex v and each neighbor n:
- **Parallel transport**: the neighbor feature `f(n)` is transported to v's frame via the precomputed `g_qp` angle, giving `f'(n)`.
- **Attention score** (gauge invariant): `score(v,n) = mean(ReLU(W_Q(f(v)) + W_K(f'(n))))`, where W_Q and W_K are equivariant RegularToRegular maps (circulant matrices). Averaging over the N group elements makes the score invariant to gauge choice.
- **Value map** (gauge equivariant, position-dependent): `W_V(u) = W_0 + W_1·u + W_2·u⊗u`, a second-order Taylor expansion in the log-map coordinate u. Each order's admissible kernel basis is solved via SVD from the equivariance constraint (eq. 78 of the paper).
- **Multi-head aggregation**: head outputs are mixed back to `in_channels` via a final RegularToRegular linear block.

#### RegularToRegular (`GERegularToRegularLinearBlock`)
Equivariant linear maps between regular fields. The equivariance constraint forces all admissible kernels to be **circulant matrices**, so the basis has exactly N elements.

#### Pooling
- **GroupPooling**: max over the N group elements — gauge invariant.
- **GlobalAveragePooling**: mean over all vertices — produces a single `channels`-dimensional descriptor.

#### ResNet structure (`GEResNetBlock`) — differences from the paper
Each block applies two MHSA layers, each with its own residual connection. Three things differ from the original paper:
- **Layer normalization**: `GELayerNorm` is applied before each attention layer (pre-norm). It normalizes jointly over channels and N, with affine parameters shared across N to preserve equivariance. The paper has no normalization.
- **Multi-head attention**: we use 2 heads; the paper does not specify this.
- **Skip connections**: each of the two attention layers inside a block has its own residual, rather than the single skip connection shown in the paper's diagram.

---

### 3. Training (`GET.py`)

- **Split**: 70% train / 15% val / 15% test, reproducible via filenumber lists saved in every checkpoint.
- **Gradient accumulation**: effective batch size controlled via `accumulation_steps` (default 16).
- **LR scheduling**: `ReduceLROnPlateau(factor=0.5, patience=5)` — reduces LR when validation loss stalls.
- **Early stopping**: monitors validation loss; stops after `patience=25` epochs without improvement.
- **Session resumption**: `load_data_from_session()` reconstructs the exact train/val/test loaders from any checkpoint; training histories are saved in checkpoints so they can be concatenated across sessions.

Current configuration: N=9, channels=12, 2 attention heads, 1 ResNet block, lr=1e-2, weight decay=1e-4, accumulation_steps=8.

**Current best result: 91% test accuracy.**

![Training results](GET/images/training.png)
