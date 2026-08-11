# How to Benchmark Models

`instantlearn benchmark` measures few-shot segmentation quality across models, backbones and datasets. It fits each model on `n_shot` reference images per category, predicts on the rest, and reports IoU.

## Setting Up Datasets

Two datasets are built in: `PerSeg` and `lvis`. Pass the directory that *contains* the dataset, not the dataset directory itself — the dataset name is appended for you.

```
/data/prompt/          <- pass this as --dataset_root
├── PerSeg/
│   ├── Images/
│   │   └── backpack/00.jpg
│   └── Annotations/
│       └── backpack/00.png
└── lvis/
```

To benchmark your own data, arrange it as described in [How to Use Custom Datasets](01-custom-dataset.md) and load it with `FolderDataset`.

## Running Benchmarks

A minimal run:

```bash
instantlearn benchmark \
  --model PerDino \
  --dataset_name PerSeg \
  --dataset_root /data/prompt \
  --class_name backpack,barn \
  --n_shot 1 \
  --device cpu \
  --overwrite
```

`--model`, `--sam` and `--dataset_name` each accept a comma-separated list or `all`, and every combination is run:

```bash
instantlearn benchmark --model Matcher,PerDino --sam SAM-HQ-tiny,SAM2-tiny --dataset_name PerSeg
```

Available models are `EfficientSAM3`, `GroundedSAM`, `Matcher`, `PerDino`, `SoftMatcher`, `SAM3-Classic` and `SAM3-Visual`.

Useful flags:

| Flag                | Purpose                                                        |
| ------------------- | -------------------------------------------------------------- |
| `--class_name`      | Preset (`default`, `benchmark`, `all`) or comma-separated names |
| `--n_shot`          | Reference images per category                                  |
| `--num_priors`      | Repeat the run, shifting which image is the reference           |
| `--precision`       | `fp32`, `fp16` or `bf16` (default `bf16`)                       |
| `--batch_size`      | Inference batch size (default 5)                               |
| `--backend`         | `pytorch` or `openvino`                                        |
| `--experiment_name` | Groups results under a named subdirectory                      |
| `--overwrite`       | Required if the output directory already exists                |
| `--save`            | Save prediction visualizations                                 |

Use `--num_priors` greater than 1 when comparing models. A single reference image makes results noisy, since one unlucky reference can dominate the score.

`bf16` is the default but is slow on CPU. Pass `--precision fp32` for CPU runs.

## Interpreting Results

Results are written to `~/outputs/`, or `~/outputs/<experiment_name>/` when `--experiment_name` is set. Each model, backbone and dataset combination also gets its own directory named `<dataset>_<backbone>_<model>`.

Two CSV files are produced. `all_results.csv` has one row per category per run:

```csv
category,iou,prior_index,inference_time,dataset_name,model_name,backbone_name
backpack,0.2860986590385437,0,0,PerSeg,PerDino,SAM-HQ-tiny
barn,0.24624915421009064,0,0,PerSeg,PerDino,SAM-HQ-tiny
```

`avg_results.csv` averages those rows per dataset, model and backbone:

```csv
dataset_name,model_name,backbone_name,category,iou,prior_index,inference_time
PerSeg,PerDino,SAM-HQ-tiny,,0.26617390662431717,0.0,0.0
```

`iou` is the intersection over union between predicted and ground-truth masks, so higher is better. `prior_index` identifies the run when `--num_priors` is greater than 1. The `category` column is empty in the averaged file because it is averaged away.

Compare models on the averaged score, then use the per-category rows to find where a model fails. A low average driven by two or three categories usually means those objects are poorly separated by the encoder, not that the model is broadly worse.

Benchmarks are not deterministic across devices. Compare runs from the same device and precision.

### Benchmarking in Python

Build a dataset and loop yourself when you want to inspect predictions rather than just scores:

```python
from instantlearn.data.torch import PerSegDataset
from instantlearn.models import PerDino

dataset = PerSegDataset(root="/data/prompt/PerSeg", categories=["backpack"], n_shots=1)
references = dataset.get_reference_dataset()
targets = dataset.get_target_dataset()

model = PerDino(device="xpu")
model.fit(references[0])

for idx in range(len(targets)):
    prediction = model.predict(targets[idx])[0]
    print(prediction.masks.shape, prediction.label_names)
```

## Known Issues

Filtering to a single category with `--class_name backpack` raises `IndexError: invalid index of a 0-dim tensor`. Pass at least two categories as a workaround.
