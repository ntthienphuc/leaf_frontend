# External training data

- `raw/` stores downloaded archives.
- `external/` stores extracted source datasets.
- `prepared/` stores generated YOLO-ready datasets.

The directories are ignored by Git through `data/.gitignore`.

## Sources

- LeavesBank: Kaggle dataset `aslhanyldrm/leavesbank-dataset`, CC BY 4.0.
- Black Pepper Leaf Disease Mini Dataset: Kaggle dataset
  `adithyantg/black-pepper-leaf-disease-mini-dataset`, CC0.

## Detector preparation

LeavesBank polygons are converted to tight one-class YOLO detection boxes. Once
the archive is extracted, run:

```powershell
python scripts/prepare_leavesbank_detect.py `
  --source data/external/leavesbank `
  --output data/prepared/leavesbank_detect
```

The detector labels are always `0: leaf`; `leaf` and `leaf_secondary` source
instances are merged. The black-pepper dataset remains a disease-classification
source because it has no leaf boxes.

Train the resulting detection dataset on Colab with:

```powershell
python scripts/train_yolo26_leaf_detect.py --device 0
```

Prepare the pepper disease images for the classifier (duplicate files are
removed by SHA-256 before splitting):

```powershell
python scripts/prepare_black_pepper_classifier.py `
  --source data/external/black_pepper_leaf_disease_mini/BLACK_PEPPER_DATASET `
  --output data/prepared/black_pepper_classifier
```
