# HUSE: Histocomponent-driven Universal Model for Virtual Immunohistochemistry Multiplex Staining via Joint Manifold Evolution (ECCV 2026)

## 1. Introduction

This repository is the official code release of **"Histocomponent-driven Universal Model for Virtual Immunohistochemistry Multiplex Staining via Joint Manifold Evolution" (HUSE, ECCV 2026)**.

HUSE is a universal "one-to-many" model that generates multiple virtual IHC biomarkers from a single H&E image. It treats the H&E image as a persistent structural scaffold via **Joint Manifold Anchoring (JMA)** — concatenating the H&E image with the noise latent as the network input — and refines heterogeneous tissue components through a **Histocomponent-driven Mixture of Experts (Hi-MoE)** with **Representation Conflict Gating (RCG)**.

## 2. News

- **2026.06.23** — Code repository created; training and evaluation code released.
- **2026.06.18** — Our paper was accepted by **ECCV 2026**! 🎉

## 3. Usage

### Environment

```bash
conda env create -f environment.yaml
conda activate huse
```

### Configuration

Each dataset has its own YAML config. The example provided here, [`configs/orion.yaml`](configs/orion.yaml), is set up for the **Orion-CRC** dataset (16 IHC markers, image size 256). To train on a different dataset, copy this file and adjust the values (e.g. `num_classes`, `data_root_*`, the marker list in `dataset_orion.py`), then point the launch script at your new YAML.

Fill in your local paths (marked with `PATH/TO/...`) before launching:

- `data_root_train` / `data_root_val`: dataset roots, organized as `<root>/he/*.png` and `<root>/<MarkerName>/*.png`.
- `num_classes`: number of biomarkers (16 for Orion-CRC).
- `clip_anchor_dir`: directory holding the offline CLIP (ViT-B/32, 1×512) anchor features used to initialize the Hi-MoE prototypes (`anchor_nuclear.pt`, `anchor_cytoplasm.pt`, `anchor_background.pt`).
- `output_dir`: where checkpoints and logs are written.

Learning-rate schedule: warmup for `warmup_epochs`, hold `lr` until `decay_start_epoch`, then cosine-decay from `lr` to `min_lr` over `[decay_start_epoch, epochs]`. With the default `epochs=400`, `decay_start_epoch=250`, the LR stays at `1e-4` until epoch 250 and then decays to `1e-5` by epoch 400.

Any YAML value can be overridden on the command line (CLI flags take priority).

### Training

The number of GPUs is set in the YAML via the `gpus` field — single card `"0"`, or multi card `"0,1,2,3"`. The launch is then a single command:

```bash
bash train.sh configs/orion.yaml
```

`output_dir`, `master_port` and `gpus` are all read from the config, so switching to another dataset/run is just a matter of pointing to a different YAML:

```bash
bash train.sh configs/your_dataset.yaml
```

> Note: `batch_size` in the YAML is the **per-GPU** batch size, so the effective batch size scales with the number of GPUs.

### Testing / Inference

Edit the checkpoint and path variables at the top of [`test.sh`](test.sh), then run:

```bash
bash test.sh
```

It generates virtual IHC images for every marker into `--output_dir` (one sub-folder per marker) and writes a PSNR / SSIM / KID report to `--output_txt`:

```bash
CUDA_VISIBLE_DEVICES=0 python test_orion.py \
    --model JiT-B/16 --img_size 256 \
    --num_sampling_steps 50 --noise_scale 2.0 --batch_size 8 \
    --checkpoint PATH/TO/checkpoint-best.pth \
    --data_path PATH/TO/Orion-CRC/test \
    --output_dir ./orion_eval_outputs \
    --output_txt ./orion_eval_outputs/metrics.txt
```

---

> More documentation (dataset construction, the mIF-to-mIHC paradigm, CLIP prototype prompts, and detailed environment setup) will be added soon.
