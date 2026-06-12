# Dataset Registry

Canonical inventory of all datasets, present and referenced. Last updated: 2026-06-12.

| Dataset | Source | Status | Train | Test (good) | Test (defect) | Known Issues |
|---------|--------|--------|-------|-------------|---------------|--------------|
| bottle | MVTec AD | complete | 209 | 20 | 63 | none |
| carpet | MVTec AD | complete | 280 | 28 | 89 | none |
| region1 | Enterprise | partial | TBD | 91 | 7 (lb=1, ps=4, py=1, tl=1) | BMP files masquerading as .png; severe class imbalance (3 classes with 1 sample each) |
| region2 | Enterprise | partial | TBD | 91 | 15 (lb=2, ps=9, py=3, tl=1) | BMP files masquerading as .png; severe class imbalance (tl has 1 sample) |
| region3 | Enterprise | mostly complete | TBD | 150 | 17 (lb=9, ps=2, py=1, tl=5) | minor imbalance (ps has 2 samples) |
| region4 | — | **MISSING** | — | — | — | Never existed on disk. README originally referenced region1-5; only 4 custom datasets exist. |
| region5 | Enterprise | partial | TBD | 91 | 23 (lb=9, ps=4, py=8, tl=2) | BMP files masquerading as .png; moderate class imbalance |

## Format Notes

- All datasets follow MVTec AD standard directory layout:
  - `train/good/` — training set (normal samples only)
  - `test/good/` — test set (normal samples)
  - `test/<defect>/` — test set (defect samples, by type)
  - `ground_truth/<defect>/` — pixel-level annotation masks
- Images in region1/2/5 have `.png` extensions but are actually BMP format. OpenCV reads them correctly via content-based format detection; current training is unaffected. If switching to an extension-based loader, convert to true PNG first.

## External Datasets

| Directory | Description |
|-----------|-------------|
| `datasets/dtd/` | Describable Textures Dataset — used by DRAEM to generate synthetic anomaly textures. Manual download required. |
