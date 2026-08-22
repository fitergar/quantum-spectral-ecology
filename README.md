# Quantum Spectral Ecological Model — Chimalapas / Ptychohyla euthysanota

This repository accompanies the article: **"Spectral modeling of the stream frog Ptychohyla euthysanota in the Chimalapas montane forest"** and contains the full computational pipeline used to produce the results reported therein.

The purpose of this work is to infer **qualitative spatial structure** of a species distribution in a remote, data-scarce environment, where field observations are extremely sparse and spatially clustered. Rather than attempting precise abundance estimation, the framework focuses on recovering **ecologically credible spatial patterns**, such as strong stream affinity and fine-scale variation along river corridors.

The approach integrates three main ingredients:

1. a probabilistic formulation of species occupancy using Gibbs measures,
2. spectral analysis of discrete Schrödinger operators defined on spatial graphs, and
3. neural networks to extrapolate inferred habitat suitability across the landscape.

---

## 1. Methodological overview and relation to the code

This section summarizes the modeling workflow at a conceptual level and explains how each methodological step is reflected in the structure of the code. Detailed file-level descriptions and execution instructions are provided in Section 2.

---

### 1.1 Spatial discretization and GIS-based preprocessing

The study region is represented as a **two-dimensional spatial grid** in UTM coordinates using the reference system **EPSG:31969 — SIRGAS 2000 / UTM zone 15N**. Each grid cell corresponds to a square of fixed side length (10 m in the original study), chosen to match the positional uncertainty of handheld GPS measurements collected during field surveys.

The GIS source layers and the four authoritative CSV exports are prepared externally. In the original study this work was performed using **QGIS**. The reproducible Python preprocessing stage then computes `drio` and `Zdrio`, aligns the 25 RGB subpixels associated with every grid cell, and writes the canonical run-specific dataset. The modeling stages consume that canonical table.

The GIS preprocessing stage produces the following spatial data products:

- a regular grid covering the study region,
- centroid coordinates for each grid cell,
- a vector representation of the river network,
- elevation data referenced to mean sea level,
- rasterized satellite imagery aligned to the grid.

Hydrological and topographic layers are obtained from **INEGI (Instituto Nacional de Estadística y Geografía)** open datasets. In particular, vector hydrography and topography correspond to the INEGI E15C55 La Gloria dataset (Serie III, 1:50 000), available at:

https://www.inegi.org.mx/app/biblioteca/ficha.html?upc=889463497363

Elevation data are derived from the INEGI **Continental Relief** products, available at:

https://www.inegi.org.mx/temas/relieve/continental/#descargas

Elevation rasters are reprojected to the project coordinate system (EPSG:31969) and sampled at grid centroids to obtain per-cell elevation values. Due to their large file size, the original elevation raster files are **not included in this repository**, but they are publicly available from INEGI and can be readily regenerated following the same procedure.

Satellite imagery is accessed through **QuickMapServices** within QGIS and rasterized onto the same grid used for spatial discretization. The resulting raster layers are aligned to the project coordinate system to ensure spatial consistency.

The repository also includes the organized pre-code spatial snapshot in `data/Gpx/GL_base_complete.gpkg`. It contains the base grid, RGB sampling layers, river geometry and river points, and field observations used to connect the external GIS preparation to the four authoritative CSV inputs. The frozen Python pipeline remains reproducible from those CSV files without QGIS.

---

### 1.2 River geometry, distance, and relative elevation

The river network plays a central role in the modeling framework, as _Ptychohyla euthysanota_ is strongly associated with stream environments. From the spatial grid and the river geometry, two key covariates are derived for each cell:

- **distance to the river**, defined as the distance from the grid cell centroid to the nearest river segment,
- **relative height with respect to the river**, capturing elevation differences between a cell and nearby river locations.

These quantities encode large-scale ecological constraints—particularly stream affinity and topographic position relative to the river—without imposing them directly in the probabilistic model.

In the original study, the river network was digitized in QGIS as a dedicated vector layer aligned to the project grid. The study region was defined by drawing a polygon enclosing the relevant section of the river corridor, and the grid was clipped to this region. River geometry was then converted into a set of reference cells or points aligned with the grid resolution, allowing distances from grid centroids to the river to be computed consistently.

Elevation values sampled at grid centroids were used to derive a relative elevation measure by comparing each cell's elevation to that of nearby river cells. This relative height is intended to capture local topographic structure along the river corridor, rather than absolute elevation above sea level.

It is important to emphasize that this GIS-based procedure is **not unique**. Distance to river and relative elevation can be computed using a variety of equivalent methods and tools, provided that the resulting quantities are defined consistently on the grid. The modeling framework does not depend on a specific implementation, only on the availability of these covariates at the grid-cell level.

Within the Python pipeline, the computation and attachment of distance-to-river and relative-elevation covariates are handled by the script `preprocess_grid_to_dataset.py`. This script takes as input a base grid CSV containing coordinates, elevation above sea level, and a river indicator, and augments it with these derived quantities in a form suitable for downstream modeling.

---

### 1.3 Satellite imagery and texture information

To capture fine-scale environmental variation along the river corridor, the framework incorporates information derived from satellite imagery. These data serve as proxies for local surface texture and habitat heterogeneity that are not captured by distance to river or elevation alone.

In the original study, satellite maps were accessed through **QuickMapServices** within QGIS and rasterized onto the same spatial grid used for discretization. Raster layers corresponding to the red, green, and blue channels were generated and reprojected to the project coordinate system (EPSG:31969). Each 10 m grid cell is represented by a consistently ordered 5-by-5 layout, giving 25 values per colour channel rather than one value per cell.

The dense RGB samples were exported to tabular form. `preprocess_grid_to_dataset.py` applies the recorded coordinate offsets, aligns the three channels by their relative subcell coordinates, and orders every cell from north to south and then west to east. The resulting `R1..R25`, `G1..G25`, and `B1..B25` columns therefore have one fixed spatial meaning throughout the dataset.

Within the Python pipeline, these RGB values are attached to each grid cell by `preprocess_grid_to_dataset.py` and subsequently combined into greyscale descriptors. The greyscale quantities are used as inputs to the neural network that modulates habitat suitability locally along the river corridor.

As with the river-related covariates, this satellite-processing workflow is **not unique**. Any raster source capable of providing spatially aligned texture information can be used, and alternative GIS or remote-sensing tools may be substituted. QuickMapServices is used solely as a convenient interface for accessing open map tiles; the modeling framework itself does not depend on a specific satellite provider, only on the availability of grid-aligned raster values exported in tabular form.

---

### 1.4 Field observations, training region, and smoothing

Field observations consist of georeferenced detections of _Ptychohyla euthysanota_ collected during a limited number of expeditions. Due to the remoteness of the study area and the logistical constraints of access, detections are extremely sparse and concentrated almost exclusively near the river network. After discretization, the majority of grid cells contain zero observations.

To stabilize inference and ensure that the inverse spectral step is numerically well behaved, the analysis is restricted to a **training region** located along the surveyed river corridor. This region corresponds to the area where field observations were collected and where the spatial support of the data is sufficiently dense to allow reliable reconstruction of the habitat-suitability potential.

In the current implementation, the training region is defined explicitly in UTM coordinates (EPSG:31969) as the union of several axis-aligned rectangular subregions that follow the sampled river corridor. A grid cell with centroid coordinates $(X, Y)$ is included in the training region if it satisfies at least one of the following conditions:

- $327634.1 < X < 327945$ and $1856536.44 < Y < 1856805.7$
- $327727.0 < X < 327946$ and $1856805.7 \le Y < 1856881.0$
- $327805.0 < X < 327946$ and $1856881.0 \le Y < 1856903.0$
- $327807.0 < X < 327946$ and $1856903.0 \le Y < 1856923.0$

The training region is further restricted to cells within a fixed maximum distance from the river: $d_{\text{rio}} \le d_{\text{max}}$.

This explicit geometric definition is encoded directly in the preparation stage of the pipeline and can be modified to accommodate different study regions or river geometries. The use of simple coordinate inequalities avoids reliance on opaque spatial joins and makes the definition of the training domain fully transparent and reproducible.

Once the training region is selected, raw detection counts (NUMPOINTS) are spatially regularized. This step is implemented in `prepare.py` and consists of applying a **Gaussian smoothing** procedure on the grid. The result is a strictly positive, continuous proxy for relative occupancy, denoted by $\psi$, defined on the training region.

The smoothing step serves two purposes:

1. it regularizes the extremely sparse observation signal implied by the field data, and
2. it ensures strict positivity of $\psi$, which is required for the inverse reconstruction of the habitat-suitability potential.

Importantly, this procedure introduces no additional ecological assumptions beyond spatial continuity at the scale of the grid. It is a numerical regularization step that enables stable spectral inference while preserving the large-scale spatial structure implied by the observations.

The output of this stage is a canonical, run-specific dataset containing the smoothed occupancy proxy and all associated covariates, which is subsequently used for potential inference, neural network training, and prediction.

---

### 1.5 Probabilistic formulation and inverse spectral inference

Species occupancy on the spatial grid is modeled as a collection of real-valued variables defined on the grid cells, with local spatial dependence encoded through nearest-neighbor interactions. This induces a Markov random field structure in which only adjacent cells interact.

Under these locality assumptions, the joint distribution of the occupancy field can be written in Gibbs form,

$$P(x) \propto \exp(-\mathcal{H}(x))$$

where the Hamiltonian $\mathcal{H}$ represents a balance between spatial smoothness and local habitat suitability.

In the model used here, the Hamiltonian consists of two components: a dispersal term that penalizes differences between neighboring cells, and a site-dependent potential that encodes habitat suitability. Together, these define a quadratic energy that can be written in operator form as

$$\mathcal{H}(x) = \langle x,(L + \mathrm{diag}(V))x\rangle$$

where $L$ is the graph Laplacian of the grid and $V$ is the environmental potential. This representation makes the connection to spectral theory explicit. The operator

$$H = L + \mathrm{diag}(V)$$

is a discrete Schrödinger operator on the grid graph, and the most probable spatial configuration of the species corresponds to its ground state.

On the training region defined in Section 1.4, the smoothed occupancy proxy $\psi$ is strictly positive. Rather than prescribing a parametric form for habitat suitability, the potential is reconstructed directly by enforcing the ground-state condition

$$(L + \mathrm{diag}(V))\psi = 0$$

This yields a pointwise expression for the potential,

$$V = -(L\psi)/ \psi$$

which uniquely determines habitat suitability on the training region under mild connectivity assumptions.

In the codebase, this inverse-spectral reconstruction is implemented using sparse linear algebra routines. Graph Laplacians and related operators are defined in `linalg.py`, while feature handling and dataset assembly are managed by utilities in `featurize.py`.

The inferred potential on the training region serves as the target variable for the neural-network models described in Section 1.6. In this way, probabilistic modeling and spectral inference provide the bridge between sparse field observations and data-driven extrapolation across the landscape.

---
### 1.6 Learning and extrapolation of habitat suitability

The environmental potential reconstructed in Section 1.5 is defined only on the training region. To extrapolate habitat suitability across the full landscape, this potential is learned as a function of environmental covariates using neural networks.

The neural-network architectures used throughout the pipeline are defined in `models.py`. This module contains the model classes that implement the distance-based driver and its locally modulated extensions, ensuring consistent input dimensionality and parameterization across training and prediction stages.

Two models are trained in sequence:

- a **distance-to-river driver**, which captures the dominant large-scale dependence of habitat suitability on proximity to the river network and is trained by `train_driver.py`,
- a **greyscale- and relative-elevation-modulated model**, which refines the driver by incorporating satellite-derived texture and relative height information near the river corridor and is trained by `train_grey.py`.

The driver model is trained first to learn the global river-distance dependence. Its parameters are then frozen, and the modulated model is trained on top of it to capture fine-scale spatial variation along the river without altering the large-scale structure.

The output of this stage is a learned, spatially continuous representation of the environmental potential that can be evaluated at any grid cell within the prediction domain.

---

### 1.7 Local spectral prediction and spatial averaging

Predicted species distributions are generated locally along the river network using the learned potential. For each river-adjacent grid cell, a square neighborhood is constructed, defining an induced subgraph of the spatial grid.

Within each neighborhood, the learned potential is evaluated, the corresponding discrete Schrödinger operator is formed, and its ground state is computed numerically. The squared amplitude of the ground state, $|\psi|^2$, is interpreted as a local probability distribution over that neighborhood.

Because neighborhoods overlap spatially, each grid cell may receive multiple local probability estimates. These local predictions are combined by simple averaging across all neighborhoods containing the cell, yielding a stable global prediction that mitigates boundary effects, scaling ambiguities, and geometric artifacts.

These prediction steps are implemented in `predict_driver_local.py` and `predict_gray_local.py`. Final prediction tables and intermediate outputs are written to the corresponding run directory under `outputs/`, enabling direct comparison between distance-only and fully modulated models.

## 2. Reproducible run protocol

### 2.1 Requirements (software + environment)

This project is a pure Python pipeline (plus PyTorch) that reads and writes CSV files. It does **not** require QGIS at runtime: all GIS work happens before the Python stage. The Python code assumes you already have the grid-level CSV inputs in `data/`.

Below are the practical requirements to run the code exactly as intended.

---

#### Python version

Use **Python 3.13**.

The locked environment uses Python 3.13 and PyTorch 2.7.1. On Linux and
Windows, PyTorch is resolved from its official CUDA 11.8 package index. This
keeps the final GPU run compatible with the NVIDIA 525.85.12 driver while
remaining usable with newer NVIDIA drivers.

---
#### Required Python packages

The core pipeline uses only the following third-party Python packages:

- **numpy**
  Core numerical computations and array handling.

- **pandas**
  CSV input/output and tabular data manipulation across all stages.

- **scipy**
  Gaussian smoothing in `prepare.py` (`scipy.ndimage`).

- **torch** (PyTorch)
  Used for:
  - Training neural-network models (`train_driver.py`, `train_grey.py`)
  - Evaluating trained models during prediction

- **numba**
  Provides the JIT decorator used by an auxiliary iterative routine in `src/linalg.py`. The frozen run uses the exact dense eigensolver described below.

All other imports in the repository come from the Python standard library.


---

#### Optional but recommended

- **CUDA-capable GPU + CUDA-enabled PyTorch**
  Training and prediction can run on CPU, but training for 100k epochs is noticeably faster with CUDA. Both `train_driver.py` and `train_grey.py` support a `--cuda` flag.
  Prediction scripts also accept `--cuda` for model evaluation (the eigen solve itself uses NumPy on the CPU).

---

#### Installation

Install `uv`, then create the exact environment recorded in `uv.lock`:

```bash
uv sync --frozen
```

Run pipeline commands through `uv run`, for example:

```bash
uv run python -m src.prepare --dataset test2 --grey-subcells
```
---

### 2.2 `src/utils.py` (paths, run directories, and naming conventions)

This repository uses a strict convention for where datasets, run artifacts, and outputs are stored. Instead of duplicating file paths inside every script, all scripts rely on a single helper module: `src/utils.py`.

The purpose of `src/utils.py` is **consistency**:
- every script reads inputs from the same locations,
- every script writes outputs using the same naming rules,
- and every run is reproducible because the output tree is predictable.

#### What `dataset_paths(dataset)` does

The most important function in `src/utils.py` is `dataset_paths(dataset)`. It takes a dataset name (for example `test2`) and returns the canonical locations for:

- the dataset CSV produced by preprocessing:
  - `data/<dataset>.csv`
- the per-run output directory:
  - `outputs/<dataset>/`
- the prepared dataset produced by the preparation stage:
  - `outputs/<dataset>/prepared.csv`

Other scripts then build on top of these canonical paths to write:
- `outputs/<dataset>/args_*.json` (saved command-line arguments)
- `outputs/<dataset>/model_*.pt` and `model_*_meta.json` (trained models + metadata)
- `outputs/<dataset>/*metrics*.csv` (training/evaluation logs)
- `outputs/<dataset>/*pred*.csv` (predictions)

This is why most scripts only ask for `--dataset <NAME>`: once the dataset name is known, every input/output file is determined automatically.

#### Helper functions you will see across scripts

`src/utils.py` also defines small helper utilities used across the pipeline. The exact set may evolve, but they typically cover:

- printing consistent status messages (progress logging)
- creating directories safely (`mkdir` behavior)
- lightweight argument saving (write `args_*.json`)
- simple error handling helpers (consistent failure messages)

These helpers are intentionally simple. The main design goal is to avoid “magic”: the repository should be runnable from the command line with predictable filenames and minimal hidden state.

#### Practical consequence for reproducibility

Because all paths are derived from `--dataset`:
- two users running the same commands with the same dataset name should produce the same output file locations,
- runs do not overwrite each other unless they reuse the same dataset name,
- and the `outputs/<dataset>/` directory contains a complete record of what was run (args files, prepared.csv, models, predictions).

This convention is what makes it possible to reproduce the results end-to-end from a clean checkout of the repository.

### 2.3 `src/preprocess_grid_to_dataset.py` — build the modeling dataset (distance/relief + RGB subpixels)

This is the **first Python step** of the pipeline. It takes:

1) a **base grid CSV** produced by GIS (cell centers + elevation + river indicator), and
2) three **RGB tile CSVs** (exported from raster sampling),

and writes a single dataset CSV to `data/<dataset>.csv` that the rest of the scripts consume.

At a high level it does two things:

- **River geometry features**
  - `drio`: distance-to-river (computed from the grid + river indicator)
  - `Zdrio`: elevation relative to the nearest river cell

- **Satellite/RGB features**
  - attaches **25 aligned RGB “subpixels”** around each grid cell:
    - `R1..R25`, `G1..G25`, `B1..B25`
  - (optional) also stores the sampled subpixel coordinates `Xsub1..Xsub25`, `Ysub1..Ysub25`

This script is intentionally “dumb but reproducible”: it does not require shapefiles or rasters at runtime.
Everything is done from CSVs so the full run can be reproduced without QGIS installed, as long as the CSV inputs exist.

---

#### Inputs

All file paths default to the `data/` directory:

- `Gridvaluesraw.csv`
  A grid table with **one row per grid cell**. At minimum it must include:
  - `X`, `Y`: cell centroid coordinates in **EPSG:31969**
  - `Z`: elevation above mean sea level (meters)
  - `Esrio`: river indicator (river cells satisfy `Esrio != 0`)

- `R2GL.csv`, `G2GL.csv`, `B2GL.csv`
  Tables with sampled raster values for each channel. Each one must include:
  - `X`, `Y`: sample coordinates
  - exactly **one value column** (the script infers which column is the pixel value)

These RGB CSVs are assumed to represent a **dense set of samples** (subpixels) around each grid cell,
so the script can pick 25 nearby samples per cell.

---

#### Outputs

- `data/<dataset>.csv`
  This is the “raw dataset” used by `prepare.py` and everything after.

The script also writes two reproducibility records under `outputs/<dataset>/`:

- `subcell_layout.csv`, which maps each feature index to its relative spatial position;
- `args_preprocess_grid_to_dataset.json`, containing the arguments, source revision,
  software environment, and input/output hashes.

---

#### What exactly gets computed?

##### 1) `drio` and `Zdrio`

A cell is considered a **river cell** if `Esrio != 0`.

For every cell, the script finds the nearest river cell using **L1 distance** in UTM coordinates:

- `drio = min_r ( |X - X_r| + |Y - Y_r| )` over river cells `r`

It also stores a river-relative elevation difference:

- `Zdrio = Z_cell - Z_nearest_river_cell`

For river cells, the nearest river cell is itself, so:

- `drio = 0` and `Zdrio = 0`

##### 2) 25 aligned RGB “subpixels” per cell: `R1..R25`, `G1..G25`, `B1..B25`

Each of the R/G/B CSVs is treated as a “cloud” of sampled raster values.

For each sample point `(x, y)`:

1. The script applies an (x, y) shift:
   - `x <- x + x_offset`
   - `y <- y + y_offset`

   This compensates for systematic coordinate offsets that can appear when exporting raster samples.

2. The shifted point is assigned to the **nearest valid grid center** `(cx, cy)`.

3. The sample is kept only if it lies within a Chebyshev (L∞) radius around the center:
   - `max(|x - cx|, |y - cy|) <= radius`

4. Within each grid cell, samples are keyed by their *relative offset* `(dx, dy)` (rounded):
   - `(dx, dy) = (x - cx, y - cy)`
   - rounded using `key_precision` decimal digits

This rounding is the key detail: it lets the script intersect the three channels (R, G, B)
so that `Rk`, `Gk`, `Bk` correspond to the **same** subpixel location.

The aligned positions are ordered explicitly from north to south and, within each
row, from west to east. Thus index 1 is the upper-left position, index 5 is the
upper-right position, and index 25 is the lower-right position in the default
5-by-5 layout. Every cell must have exactly the same relative layout or preprocessing
stops with an error. The script writes:

- `R1..R25`, `G1..G25`, `B1..B25`

If `--save-subcoords` is enabled, it also writes:

- `Xsub1..Xsub25`, `Ysub1..Ysub25`

These are the (shifted) subpixel coordinates corresponding to the chosen samples.

---

#### Command-line usage

Basic run (using defaults in `data/`):

```bash
python -m src.preprocess_grid_to_dataset \
  --dataset test2
```
#### Reproducibility note (exact frozen parameters)

The recommended way to reproduce the run exactly is to use the argument snapshot saved during the final run:

- `outputs/test2/args_preprocess_grid_to_dataset.json`

This file contains the complete set of parameters (including alignment offsets, matching precision, and sampling radius) used to generate the frozen dataset `data/test2.csv`.

In the README we show a minimal command for convenience, but the JSON above is the authoritative record of the final preprocessing configuration.

(Analogously, the full run is documented by the other `outputs/test2/args_*.json` files for the subsequent stages: preparation, training, and prediction.)

### 2.4 `src/prepare.py` — compute greyscale + define training region + smooth observations

This is the **second Python step** of the pipeline. It reads `data/<dataset>.csv` (produced by preprocessing) and writes a run-specific prepared table to:

- `outputs/<dataset>/prepared.csv`

Conceptually, `prepare.py` is where the pipeline turns “raw covariates + sparse point counts” into a **stable training target** on a **well-defined training region**.

It performs three main tasks:

1) compute a greyscale feature from the RGB subcell values (and optionally keep per-subcell greys),
2) define the training region explicitly in UTM coordinates (plus a `drio` cutoff),
3) smooth sparse field counts (`NUMPOINTS`) using a Gaussian smoothing scheme on the training region only.

This script is where you *must* modify the training region if you change study area geometry.

---

#### Inputs

- `data/<dataset>.csv`
  Produced by `preprocess_grid_to_dataset.py`. This file must already contain:
  - `X`, `Y` (grid cell centroid coordinates)
  - `NUMPOINTS` (raw counts per grid cell; usually 0 for most cells)
  - `drio` (distance-to-river) if you want the `--max-drio` cutoff to apply
  - `R1..R25`, `G1..G25`, `B1..B25` (if you want greyscale features)

---

#### Outputs

- `outputs/<dataset>/prepared.csv`
  This file contains the original dataset columns plus new derived columns:
  - `Grey` (if RGB triplets exist)
  - optionally `Grey1..GreyN` if `--grey-subcells` is enabled
  - `<vcol>_SMOOTH` (by default `NUMPOINTS_SMOOTH`) if smoothing is enabled

The script also writes:

- `outputs/<dataset>/args_prepare.json`
  This JSON captures the full command line and resolved parameters for the run.

---

#### What exactly gets computed?

##### 1) Greyscale features (`Grey` and optional `Grey1..GreyN`)

`prepare.py` detects how many RGB triplets exist in the dataset (up to 25). If `Rk`, `Gk`, `Bk` exist for `k=1..N`, then `N` triplets are found.

For each subcell triplet, the script computes a standard luminance greyscale value:

- `Greyk = 0.299*Rk + 0.587*Gk + 0.114*Bk`

If you enable `--grey-subcells`, these per-subcell values are kept as separate columns:
- `Grey1..GreyN`

In all cases, it also computes one aggregated greyscale descriptor called `Grey` using a weighted formula controlled by `--gamma`:

- first normalize each cell’s subcell greys to `[0,1]` using per-cell min/max
- then compute a weighted average that emphasizes darker (or brighter) components depending on `gamma`
- finally scale to `[0,100]`

In plain terms:
- `Grey` is a **single per-cell texture descriptor** derived from the 25 subcell samples,
- `--gamma` controls how strongly the aggregation emphasizes extremes (default `3.0`).

If the dataset does not include any RGB triplets, greyscale computation is skipped.

---

##### 2) Training region definition (explicit corridor mask + distance-to-river cutoff)

The training region is not inferred automatically. It is explicitly defined in `prepare.py` by a coordinate mask in UTM (EPSG:31969).

The region is a union of axis-aligned rectangular blocks that follow the sampled river corridor. A cell is inside the region if `(X, Y)` satisfies at least one of the following:

- $327634.1 < X < 327945$ and $1856536.44 < Y < 1856805.7$
- $327727.0 < X < 327946$ and $1856805.7 \le Y < 1856881.0$
- $327805.0 < X < 327946$ and $1856881.0 \le Y < 1856903.0$
- $327807.0 < X < 327946$ and $1856903.0 \le Y < 1856923.0$

In addition, if the dataset contains a `drio` column, the region is further restricted by:

- `drio <= --max-drio` (default `80.0`)

This makes the training region definition fully reproducible and easy to edit. For a new study site, you should update these inequalities (and document them in the README).

---

##### 3) Smoothing sparse counts (`NUMPOINTS`) on the training region only

Field detections are extremely sparse; most cells have `NUMPOINTS = 0`. To obtain a stable occupancy proxy for inverse inference, the script computes a smoothed version of the count column.

The default smoothing method is `--smooth step`, which performs a “step-embedding” Gaussian filter:

1) place the training-region counts into a coarse grid array `u`
2) upsample the grid by factor `--s` (default `100`) to create a fine grid
3) apply a Gaussian blur on the fine grid
4) downsample back to the coarse grid by averaging

The Gaussian width is specified in meters:
- `--sigma-meters` (default `8.0`)

and converted to pixel units using:
- `--cellsize` (default `10.0`) and `--s`

After smoothing, the script renormalizes the result so that total mass is preserved on the training region:

- `sum(NUMPOINTS) == sum(NUMPOINTS_SMOOTH)` on training cells

Finally (important implementation detail): the smoothed values are restricted to the **true training cells only** (no rectangular spillover). Cells outside the training region remain `NaN` in the smooth column after merging into the full dataset.

The smoothing output column is named:
- `<vcol>_SMOOTH` (default `NUMPOINTS_SMOOTH`)

---

#### Command-line usage

Typical run (recommended):

```bash
python -m src.prepare \
  --dataset test2 \
  --grey-subcells \
  --smooth step \
  --sigma-meters 8.0 \
  --cellsize 10.0 \
  --s 100 \
  --max-drio 80.0
```
### 2.4 Training the habitat-suitability models

This stage trains the neural network models used to extrapolate habitat suitability across the landscape. Training is performed in **two sequential steps** and must be executed in order.

The core idea is:
1. first learn a **distance-to-river baseline model**,
2. then **freeze that model** and train a second network that locally refines it using satellite texture and elevation.

---

#### 2.4.1 Model definitions (`src/models.py`)

All neural network architectures used in training and prediction are defined in `src/models.py`. This module contains the model classes and ensures that the same architectures are used consistently during training and prediction.

There are two relevant model types:
- a **driver model**, which depends only on distance to the river,
- a **modulated model**, which multiplies the driver output by a locally gated correction based on greyscale satellite information and relative elevation.

The modulated model internally holds a copy of the trained driver and assumes it is frozen.

---

#### 2.4.2 Training the distance-to-river driver (`src/train_driver.py`)

The first training step fits the **driver model**.

Key points:
- Input data are read from `outputs/<dataset>/prepared.csv`.
- The training target is the inferred potential computed from the prepared dataset.
- Only the driver model parameters are optimized at this stage.
- Training writes:
  - trained model weights (driver),
  - model metadata,
  - training metrics,
  - diagnostic prediction tables.

The full set of parameters used for this stage is saved to:
- `outputs/<dataset>/args_train_driver.json`

**Example run (recommended):**
```bash
python -m src.train_driver \
  --dataset test2 \
  --seed 0 \
  --cuda
```

For the exact configuration used in the frozen reference run, see:
`outputs/test2/args_train_driver.json`

---

#### 2.4.3 Training the greyscale-modulated model (`src/train_grey.py`)

The second training step fits the modulated model and must be run after the driver has been trained.

Key points:
- The previously trained driver model is loaded and frozen (its weights are not updated).
- The modulated model learns a multiplicative correction that depends on:
  - greyscale satellite features, and
  - relative elevation with respect to the river.
- The modulation is activated only near the river; far from the river the model reduces to the driver.
- Training writes:
  - trained model weights (modulated),
  - model metadata,
  - training metrics,
  - diagnostic prediction tables.

The full set of parameters used for this stage is saved to:
- `outputs/<dataset>/args_train_grey.json`

**Example run (recommended):**
```bash
python -m src.train_grey \
  --dataset test2 \
  --seed 0 \
  --cuda
```

For the exact configuration used in the frozen reference run, see:
`outputs/test2/args_train_grey.json`

---

#### 2.4.4 Supporting modules (`featurize.py` and `linalg.py`)

Two supporting modules are used internally by both training scripts:

**`src/featurize.py`**
Handles feature assembly, normalization, and construction of training tensors from `prepared.csv`.

**`src/linalg.py`**
Provides graph-based operators and numerical routines required by the training pipeline.

These modules are not called directly by the user. They exist to keep the training scripts short, consistent, and reproducible.

---

#### 2.4.5 Execution order (summary)

The correct execution order is:
1. `python -m src.train_driver --dataset <dataset>`
2. `python -m src.train_grey --dataset <dataset>`

Running the second script without first training the driver will fail.

Each stage records its full configuration in `outputs/<dataset>/args_*.json`, and these files should be used as the reference when reproducing the frozen results.

### 2.5 Local prediction (neighborhoods + spectral solve + averaging)

Prediction is performed locally along the river corridor using overlapping square neighborhoods. There are three scripts involved and they must be run in this order:

1. `src/neighborhoods.py` (build neighborhoods along the river)
2. `src/predict_driver_local.py` (predict using the distance-to-river driver model)
3. `src/predict_gray_local.py` (predict using the greyscale+elevation modulated model)

Both prediction scripts load the trained models, evaluate the learned potential on each neighborhood, compute a local ground state, and then average local probabilities across overlapping neighborhoods.

---

#### 2.5.1 Build neighborhoods (`src/neighborhoods.py`)

This script creates the list of overlapping neighborhoods used for prediction and saves them as a pickle file:

- `outputs/<dataset>/VecindadesLinf.pkl`

Neighborhood construction:

- Neighborhood centers are chosen as the cells with `drio ≈ 0` (river-center cells).
- For each river cell, a square neighborhood is defined using Chebyshev distance (L∞):
  - a cell is included if `max(|dx|, |dy|) <= half_size`
- The default `half_size=60.0` meters produces neighborhoods that correspond to a 120 m × 120 m square.

Saved configuration:

- `outputs/<dataset>/args_neighborhoods.json`

**Example run:**
```bash
python -m src.neighborhoods \
  --dataset test2 \
  --half-size 60.0 \
  --drio-center 0.0 \
  --drio-tol 1e-9
```

---

#### 2.5.2 Driver-only prediction (`src/predict_driver_local.py`)

This script generates predictions using only the distance-to-river driver model trained in Section 2.4.

Inputs:

- `outputs/<dataset>/prepared.csv` (from `prepare.py`)
- `outputs/<dataset>/VecindadesLinf.pkl` (from `neighborhoods.py`)
- `outputs/<dataset>/model_driver.pt` (from `train_driver.py`)

Outputs:

- `outputs/<dataset>/pred_driver_localavg.csv` (default name; may vary by args)
- `outputs/<dataset>/args_predict_driver_local.json`

What it does (high level):

For each neighborhood:

- builds the model input tensor for the neighborhood (internally via `featurize.py`)
- evaluates the trained driver to obtain a per-cell potential value
- constructs a neighborhood Laplacian and forms the local operator
- computes a local ground state numerically
- converts it into a local probability and accumulates it into a global average

After all neighborhoods:

- averages probabilities by the number of times each cell was seen
- normalizes the final probability field
- optionally calibrates to population totals if a training count column is available

Only cells contained in at least one neighborhood are written to the prediction table. Thus the prediction domain is the union of the local river-corridor neighborhoods, not necessarily the entire input grid. In the frozen `test2` run, this domain contains 1,942 of the 2,890 grid cells. Population calibration is computed on the overlap between this prediction domain and cells with a valid smoothed training signal.

**Example run:**
```bash
python -m src.predict_driver_local \
  --dataset test2 \
  --neigh-file VecindadesLinf.pkl \
  --out-csv pred_driver_localavg.csv
```

---

#### 2.5.3 Greyscale+elevation prediction (`src/predict_gray_local.py`)

This script generates predictions using the full modulated model (driver + local greyscale/elevation correction).

Inputs:

- `outputs/<dataset>/prepared.csv`
- `outputs/<dataset>/VecindadesLinf.pkl`
- `outputs/<dataset>/model_driver.pt`
- `outputs/<dataset>/model_grey.pt`

Outputs:

- `outputs/<dataset>/pred_grey_localavg.csv` (default)
- `outputs/<dataset>/args_predict_grey_local.json`

Important behavior:

- The driver is loaded first and immediately frozen.
- The greyscale model is then loaded and used to evaluate the full potential on each neighborhood.
- Neighborhood predictions are combined by averaging, just like in the driver-only case.
- This script also prints optional convergence diagnostics during the local solve if enabled by args.

**Example run:**
```bash
python -m src.predict_gray_local \
  --dataset test2 \
  --neigh-file VecindadesLinf.pkl \
  --out-csv pred_grey_localavg.csv
```

---

#### 2.5.4 Supporting modules used internally (`featurize.py` and `linalg.py`)

Both prediction scripts rely on two supporting modules:

**`src/featurize.py`**
Builds the normalized input tensor for each neighborhood. In particular, the function `build_Tdata_grey25(...)` constructs the per-cell feature matrix used by both the driver and greyscale models (the driver reads only its distance-to-river channel, but uses the same tensor).

**`src/linalg.py`**
Provides the grid Laplacian construction (`laplacian_4nbrs`) and the direct
symmetric ground-state solver (`ground_state_symmetric`).

These modules are not called directly from the command line; they are used internally to keep prediction code consistent and reproducible.

---

#### 2.5.5 Execution order (summary)

A full prediction run follows this order:

1. `python -m src.neighborhoods --dataset <dataset>`
2. `python -m src.predict_driver_local --dataset <dataset>`
3. `python -m src.predict_gray_local --dataset <dataset>`

Each stage writes an `outputs/<dataset>/args_*.json` file that records the exact parameters used.

### 2.6 Exact reproduction (end-to-end, same order as the paper run)

This repository saves the command-line arguments used in the final run as JSON files under:
- `outputs/<dataset>/args_preprocess_grid_to_dataset.json`
- `outputs/<dataset>/args_prepare.json`
- `outputs/<dataset>/args_train_driver.json`
- `outputs/<dataset>/args_train_grey.json`
- `outputs/<dataset>/args_neighborhoods.json`
- `outputs/<dataset>/args_predict_driver_local.json`
- `outputs/<dataset>/args_predict_grey_local.json`

For the frozen reference run in this repository (`dataset = test2`), the expected execution order is:

1) preprocessing → 2) prepare → 3) train driver → 4) train grey → 5) build neighborhoods → 6) predict (driver) → 7) predict (grey)

The frozen artifacts are stored under `outputs/test2/`. `SOURCE_COMMIT` records the clean repository revision containing the numerical source used for the run, the `args_*.json` files record every effective argument and runtime, and `SHA256SUMS` verifies the generated dataset and complete output directory.

The complete `test2` run is retained as the frozen baseline. After that run was verified, the greyscale/elevation model alone was retrained with a stronger but still bounded modulation. This selected refinement is stored under `outputs/test2-grey-radical1/`; it reuses the frozen `prepared.csv`, driver checkpoint, driver metadata, and neighborhoods from `test2`. Its own `SOURCE_COMMIT`, argument records, model metadata, diagnostics, predictions, and `SHA256SUMS` preserve the provenance of the selected result. The final GIS package and generated figures combine the target potential and driver prediction from `test2` with the selected grey prediction from `test2-grey-radical1`.

Below is an explicit command sequence that reproduces the full pipeline.

**Important:** The scripts have many optional flags; the *exact* values used are recorded in the JSON files above. If your local defaults differ from the saved configuration, use the JSON files as the reference and pass the corresponding flags explicitly.

---

#### 2.6.1 Run everything (example: `test2`)
```bash
# (0) Optional sanity check: confirm you are in repo root
# ls should show: data/ src/ outputs/ README.md

# (1) Build dataset from GIS-exported CSVs (adds drio, Zdrio, RGB subcells)
python -m src.preprocess_grid_to_dataset --dataset test2

# (2) Prepare training table (defines training region internally + greyscale + smoothing)
python -m src.prepare --dataset test2 --grey-subcells

# (3) Train distance-to-river driver model (must run before train_grey)
python -m src.train_driver --dataset test2 --seed 0 --cuda

# (4) Train greyscale/elevation modulated model (loads + freezes the driver)
python -m src.train_grey --dataset test2 --seed 0 --cuda

# (5) Build overlapping neighborhoods along the river
python -m src.neighborhoods --dataset test2

# (6) Predict with driver-only model (local spectral solve + averaging)
python -m src.predict_driver_local --dataset test2

# (7) Predict with full greyscale/elevation modulated model (local spectral solve + averaging)
python -m src.predict_gray_local --dataset test2
```

After a successful run, the main artifacts you should see are:

- `data/test2.csv`
  Output of preprocessing (grid + `drio`, `Zdrio`, RGB subcell columns).

- `outputs/test2/prepared.csv`
  Output of preparation (training mask applied internally, greyscale features, and smoothed counts).
  Note that to get the training region one just has to select rows for which the column with the smoothed
  values is not NAN.

- `outputs/test2/model_driver.pt` + `outputs/test2/model_driver_meta.json`
  Trained **driver** network weights + metadata.

- `outputs/test2/model_grey.pt` + `outputs/test2/model_grey_meta.json`
  Trained **greyscale/elevation modulated** network weights + metadata.

- `outputs/test2/train_driver_predictions.csv`
  Driver training diagnostics table. This file includes the **target potential values** (reconstructed on the training region) and the **driver model’s predicted potential** for the same cells.

- `outputs/test2/train_grey_predictions.csv`
  Greyscale-model training diagnostics table. This file includes the same **target potential** and the **full modulated model’s predicted potential** (driver + local correction).

- `outputs/test2/train_driver_metrics.csv`
  Metrics logged during driver training.

- `outputs/test2/train_grey_metrics.csv`
  Near-river and full-region metrics for the greyscale/elevation model.

- `outputs/test2/VecindadesLinf.pkl`
  Neighborhood list (pickle) used by the local spectral prediction stage.

- `outputs/test2/pred_driver_localavg.csv`
  Final **driver-only** local spectral prediction averaged across neighborhoods.

- `outputs/test2/pred_grey_localavg.csv`
  Baseline **full model** (driver + greyscale/elevation) local spectral prediction averaged across neighborhoods.

#### 2.6.2 Selected greyscale refinement

The selected final greyscale/elevation result was generated from the frozen baseline prerequisites with:

```bash
mkdir -p outputs/test2-grey-radical1
cp outputs/test2/prepared.csv outputs/test2-grey-radical1/
cp outputs/test2/model_driver.pt outputs/test2-grey-radical1/
cp outputs/test2/model_driver_meta.json outputs/test2-grey-radical1/
cp outputs/test2/VecindadesLinf.pkl outputs/test2-grey-radical1/

uv run --frozen python -m src.train_grey \
  --dataset test2-grey-radical1 \
  --seed 0 \
  --cuda \
  --mod-poly 2 \
  --a 0.0 \
  --b 2.0 \
  --d0 30 \
  --k 0.5 \
  --epochs 100000 \
  --lr0-inv 50 \
  --step-size 0.5 \
  --log-every 2500

uv run --frozen python -m src.predict_gray_local \
  --dataset test2-grey-radical1
```

The authoritative effective configuration is recorded in `outputs/test2-grey-radical1/args_train_grey.json` and `model_grey_meta.json`. The selected final prediction is `outputs/test2-grey-radical1/pred_grey_localavg.csv`.

#### Visualizing results in GIS or other tools

The GIS and figure stages are also frozen and scripted. They use **QGIS** and the following files:

- `data/Gpx/GL_base_complete.gpkg`: styled pre-code spatial layers.
- `data/Gpx/build_qgis_package.py`: joins the frozen CSV results to the base grid and creates `GL_test2_radical1_results.gpkg`.
- `data/Gpx/GL_test2_radical1_results.gpkg`: self-contained styled result layers for the study grid, smoothed observations, inverse potential, driver prediction, and selected grey prediction, together with the supporting spatial layers.
- `data/Gpx/export_qgis_figures.py`: creates the five persistent QGIS layouts, saves `GL_test2_radical1_figures.qgz`, and exports the corresponding PNG and PDF figures.
- `figures/generated/`: frozen exports, `figure_manifest.json`, and `SHA256SUMS`.

Run each script from the QGIS Python Console, replacing the repository path as needed:

```python
from pathlib import Path

PROJECT_ROOT = Path("/absolute/path/to/quantum-spectral-ecology").resolve()
exec((PROJECT_ROOT / "data/Gpx/build_qgis_package.py").read_text())
exec((PROJECT_ROOT / "data/Gpx/export_qgis_figures.py").read_text())
```

`build_qgis_package.py` deliberately refuses to replace an existing result package while `ALLOW_OVERWRITE = False`; enable replacement only when intentionally regenerating it. The figure exporter uses a live Google Satellite XYZ layer as the cartographic background. That external basemap is not stored in the GeoPackage, whereas the exported PNG and PDF figures are frozen and checksummed.

The CSV results can also be visualized with any equivalent tool that supports spatial data, for example:

- **Python** (GeoPandas / Matplotlib)
- **R** (sf / ggplot2)
- other GIS software capable of reading CSV point layers

In all cases, `X` and `Y` must be interpreted as **EPSG:31969** coordinates.
