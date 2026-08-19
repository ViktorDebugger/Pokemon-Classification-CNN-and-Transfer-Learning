# Pokemon Classification: CNN and Transfer Learning

The project classifies Pokemon images using PyTorch: custom convolutional architectures (comparing Global Average Pooling against Flatten) and transfer learning on pretrained models (ResNet18, MobileNetV2).

All code and training live in `workspace/notebook.ipynb`; the environment is reproduced via Docker + `uv`.

## Requirements

- **An NVIDIA GPU is mandatory.** `docker-compose.yml` reserves a GPU device unconditionally (`deploy.resources.reservations.devices`) — without an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on the host, `docker compose up` fails immediately and the container will not start. There is no CPU fallback.
- A host NVIDIA driver compatible with CUDA 12.8 (roughly version 550+).
- Linux/amd64 or Windows with WSL2. The `nvidia/cuda` base image is amd64-only, so this will not build or run on Apple Silicon (M1/M2/M3) or other ARM hosts.
- Docker and Docker Compose.

## Project structure

```
.
├── Dockerfile              # CUDA 12.8 + Python 3.12 + uv
├── docker-compose.yml      # pokemon-classifier service, GPU passthrough, volumes
├── .gitignore
├── README.md
└── workspace/
    ├── .vscode/
    │   └── settings.json   # excludes .venv from VS Code's watcher/search
    ├── dataset/             # images per class (ImageFolder format)
    │   ├── combee/
    │   ├── comfey/
    │   └── ...
    ├── models/              # saved model weights after training (*.pt)
    ├── notebook.ipynb        # code, training and visualizations
    ├── pyproject.toml        # dependencies (torch/torchvision from CUDA index, jupyter, etc.)
    └── uv.lock                # pinned dependency versions
```

## Dataset

Uses the `torchvision.datasets.ImageFolder` format — each class in its own subfolder under `workspace/dataset/`:

```
dataset/
├── combee/
├── comfey/
├── conkeldurr/
└── ...
```

Train/validation split — **80/20** (`VAL_RATIO = 0.2`, fixed `SEED = 42`), computed automatically from however many images currently live in `dataset/` when the notebook runs. The notebook itself prints the number of classes and split sizes right after loading the data — they aren't hardcoded here since they depend on what's currently in `dataset/`.

## Setup and running (Docker + uv)

1. Build and start the container:
   ```
   docker compose up --build -d
   ```
2. Install the dependencies inside the container — this step is **required on first run** and is not automatic (`docker-compose.yml` sets `command: bash`, which overrides the Dockerfile's `CMD`):
   ```
   docker compose exec pokemon-classifier bash
   uv sync
   ```
   `uv sync` installs `torch`/`torchvision` from the CUDA build plus the rest of the dependencies into `/workspace/.venv`. The shell already starts in `/workspace` (the image's `WORKDIR`). The first sync takes a while — the CUDA wheels are large.
3. Attach to the container via VS Code:
   - Install the **Dev Containers** extension.
   - `Ctrl+Shift+P` → **"Dev Containers: Attach to Running Container..."** → select `/pokemon-classifier`.
   - In the new window, open the `/workspace` folder.
4. Open `notebook.ipynb`, select the `/workspace/.venv/bin/python` kernel, and run the cells top to bottom.

The virtual environment lives in the named volume `venv-data`, not in `./workspace`, so it survives container restarts — `uv sync` only needs to be repeated when `pyproject.toml` changes. To wipe it and start from a clean environment, run `docker compose down -v`.

On first run, the transfer learning cells will automatically download pretrained ResNet18 and MobileNetV2 weights (ImageNet).

## What's implemented

- Data loading via `ImageFolder` and `DataLoader`
- A custom CNN with three convolutional blocks (Conv → BatchNorm → ReLU → MaxPool)
- Two classifier head variants:
  - **CNNFlatten** — classic `Flatten` before the fully-connected layer
  - **CNNGAP** — `Global Average Pooling` before the classifier
- Training with `Adam` and a `StepLR` scheduler, tracking the best checkpoint by val accuracy
- Transfer learning on **ResNet18** and **MobileNetV2**:
  - **Feature extraction** — only the new classifier head is trained
  - **Fine-tuning** — the last convolutional blocks are additionally unfrozen
- Comparison across all 6 experiments: accuracy, precision, recall, F1, training time, loss/accuracy curves, an annotated confusion matrix, and a summary markdown table with the best metrics highlighted

## Compared models

| Model | Input size | Description |
|-------|------------|--------------|
| CNNFlatten | 32×32 | Custom CNN with Flatten |
| CNNGAP | 32×32 | Custom CNN with Global Average Pooling |
| ResNet18 FE | 224×224 | Feature extraction |
| ResNet18 FT | 224×224 | Fine-tuning layer4 + fc |
| MobileNetV2 FE | 224×224 | Feature extraction |
| MobileNetV2 FT | 224×224 | Fine-tuning last block + classifier |

Results (accuracy, precision/recall/F1, training time) are generated automatically by the notebook in the "Summarize results" and "Summary statistics: best model per metric" cells — they depend on the current contents of `dataset/`, so they aren't hardcoded here (to avoid going stale).

## Notes

- Custom CNN images are resized to 32×32; transfer learning images to 224×224 with ImageNet normalization.
- Training runs on the GPU (`cuda`) — required, see Requirements above. Verify it's picked up correctly with the `torch.cuda.is_available()` cell in the notebook.
- Default hyperparameters: 10 epochs, batch size 32, learning rate `1e-3` (`1e-4` for fine-tuning).
- Trained model weights (`workspace/models/*.pt`) are not committed to git — see `.gitignore`.
