# Mixed INT4/INT8 Quantization for ECG Classification

A compact, reproducible reference project for **mixed-precision INT4/INT8
quantization-aware training (QAT)** on a one-dimensional CNN for MIT-BIH ECG
heartbeat classification. This repository is designed for Computer Engineering
students interested in efficient deep learning and embedded AI deployment.

> [!IMPORTANT]
> This project is intended for research and education only. It is not a medical
> device and must not be used for clinical diagnosis.

## Project overview

The project applies sensitivity-guided mixed-precision quantization to a compact
PaperInceptionCNN model. The 40% least-sensitive quantized layers use INT4
weights, while the remaining layers use INT8 weights. Activations use 8-bit
fake quantization.

The pipeline includes:

- MIT-BIH heartbeat extraction and per-beat normalization;
- a depthwise-separable Inception-style 1D CNN;
- layer-wise sensitivity-based bit-width assignment;
- fake-quantized and real-quantized Python reference models;
- quantization-range, observer-freezing, size-accounting, and parity tests;
- reproducible checkpoints, configuration files, and evaluation reports.

## Results

| Metric | Result |
|---|---:|
| Architecture | PaperInceptionCNN with depthwise-separable Conv1d |
| Batch normalization | Not used |
| Parameters | 1,658 |
| Input | One ECG beat, `1 x 260` samples |
| AAMI classes | 5 (`F`, `N`, `Q`, `S`, `V`) |
| Quantized layers | 12 INT4 and 19 INT8 |
| FP32 accuracy | 98.33% |
| Mixed INT4/INT8 accuracy | 97.76% |
| Mixed INT4/INT8 macro F1 | 87.63% |
| FP32 parameter size | 6,632 bytes |
| Estimated mixed packed size | 3,587 bytes |
| Estimated compression ratio | 1.85x |

Complete machine-readable results are available in
[`results/mixed_precision_report.json`](results/mixed_precision_report.json).

### Evaluation warning

The published checkpoints were produced using a stratified random beat-level
split rather than a patient-disjoint split. Beats from the same patient may
therefore occur in both training and test sets. These results must not be
interpreted as evidence of generalization to unseen patients.

## Repository structure

```text
.
├── config/                  # Sensitivity data used for bit-width selection
├── data/                    # Dataset download and preparation instructions
├── docs/                    # Architecture, audit, and portfolio documentation
├── models/                  # FP32 and mixed-QAT checkpoints
├── results/                 # Published reference evaluation reports
├── scripts/                 # Command-line inspection utilities
├── src/                     # Model, QAT, and real-quant reference code
├── tests/                   # Dataset-independent regression tests
└── .github/workflows/       # Continuous integration configuration
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the processing pipeline
and [`docs/REPOSITORY_AUDIT.md`](docs/REPOSITORY_AUDIT.md) for the repository
quality audit.

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- NumPy
- scikit-learn
- WFDB
- tqdm

## Quick start

Clone the repository and create a virtual environment:

```bash
git clone <YOUR_REPOSITORY_URL>
cd DLSYS_MIX_INT4_INT8
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the dataset-independent regression tests:

```bash
python -m unittest discover -s tests -v
```

Inspect the model architecture and parameter distribution:

```bash
python scripts/inspect_model.py
```

## Dataset preparation

Download the MIT-BIH Arrhythmia Database by following
[`data/README.md`](data/README.md). Each record must contain matching `.hea`,
`.dat`, and `.atr` files.

The local dataset directory is excluded by `.gitignore` and must not be committed
to the repository.

## Evaluating the checkpoint

Set the dataset path and run the mixed-precision pipeline.

Windows PowerShell:

```powershell
$env:MITBIH_DATASET_PATH = "D:\datasets\mitdb"
python src/mixed_precision.py
```

Linux or macOS:

```bash
MITBIH_DATASET_PATH=/path/to/mitdb python src/mixed_precision.py
```

If `MITBIH_DATASET_PATH` is not set, the program looks for the dataset in
`data/mitdb/`. Newly generated reports are written to `results/generated/` so
that the published reference results are not overwritten.

## Quantization methodology

Weights use symmetric per-channel quantization. Activations use asymmetric
per-tensor quantization. Layer bit widths are selected from the leave-one-out
sensitivity results stored in `config/int8_sensitivity.json`.

Before test evaluation, activation observers are reset, calibrated using only
training samples, and then frozen. This prevents test-set statistics from
updating the quantization parameters.

The real-quantized Python reference stores actual integer weight values:

- INT4 layers are restricted to `[-8, 7]`;
- INT8 layers are restricted to `[-128, 127]`;
- quantized weights are dequantized before PyTorch convolution or linear
  operations.

## Deployment limitations

The current implementation is a Python reference, not an integer-only embedded
runtime. INT4 values are stored in `torch.int8` tensors and are not yet packed as
two 4-bit nibbles per byte. Consequently, the reported mixed-model size is a
**packed deployment estimate**, not the physical size of the Python buffers.

The repository does not currently include:

- packed INT4 C kernels;
- bit-exact Python-to-C validation;
- ARM CMSIS-NN or microcontroller deployment;
- measured latency, RAM, Flash, or energy consumption;
- patient-disjoint DS1/DS2 evaluation.

These limitations are documented explicitly to keep the reported results
technically accurate and reproducible.

## Suggested future work

- retrain and evaluate using a patient-disjoint DS1/DS2 protocol;
- implement packed INT4 storage and C inference kernels;
- validate Python and C outputs using golden test vectors;
- deploy the model on an ARM Cortex-M target;
- measure execution time, Flash, SRAM, and energy consumption;
- compare full INT8, mixed INT4/INT8, pruning, and FP32 baselines.

## Using this project in a portfolio

When describing the project in a CV, state what you personally implemented and
include the evaluation protocol—not only the highest accuracy value. An example
description and a publication checklist are provided in
[`docs/PORTFOLIO_GUIDE.md`](docs/PORTFOLIO_GUIDE.md).

## Reproducibility and testing

The regression suite checks:

- INT4 and INT8 integer ranges;
- parity between frozen fake quantization and real-quant reference execution;
- observer freezing during validation;
- consistent raw and packed model-size accounting;
- rejection of incomplete MIT-BIH records.

GitHub Actions compiles the Python sources and runs these tests automatically
for every push and pull request.

## License and data attribution

The source code is released under the [MIT License](LICENSE).

The MIT-BIH Arrhythmia Database is distributed separately by PhysioNet. Users
must download it themselves and comply with the dataset terms and citation
requirements. No MIT-BIH signal or annotation files are included in the Git
upload set.
