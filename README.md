# HUSE: Histocomponent-driven Universal Model for Virtual Immunohistochemistry Multiplex Staining via Joint Manifold Evolution (ECCV 2026)

## 1. Introduction

This repository is the official code release of **"Histocomponent-driven Universal Model for Virtual Immunohistochemistry Multiplex Staining via Joint Manifold Evolution" (HUSE, ECCV 2026)**.

HUSE is a universal "one-to-many" model that generates multiple virtual IHC biomarkers from a single H&E image. It treats the H&E image as a persistent structural scaffold via **Joint Manifold Anchoring (JMA)** — concatenating the H&E image with the noise latent as the network input — and refines heterogeneous tissue components through a **Histocomponent-driven Mixture of Experts (Hi-MoE)** with **Representation Conflict Gating (RCG)**.

## 2. News

- **2026.09.30** — Added the ORION-CRC mIF-to-mIHC dataset preparation pipeline.
- **2026.09.30** — Released the CLIP anchor features for the three Hi-MoE experts: nuclear, cytoplasm, and background.
- **2026.06.23** — Code repository created; training and evaluation code released.
- **2026.06.18** — Our paper was accepted by **ECCV 2026**! 🎉

## 3. Usage

### Environment

```bash
conda env create -f environment.yaml
conda activate huse
```

### Dataset Preparation

HUSE uses the paired H&E and mIF tiles from the processed **ORION-CRC** dataset released by the [MIPHEI-ViT](https://github.com/sanofi-public/miphei-vit) authors.

1. Download [`ORIONCRC_dataset_tile_20x.zip`](https://zenodo.org/records/15340874/files/ORIONCRC_dataset_tile_20x.zip?download=1) from the [MIPHEI-ViT Zenodo record](https://doi.org/10.5281/zenodo.15340874). This is the approximately 127 GB archive containing the paired `he/` and `if/` tiles. The similarly named `ORIONCRC_dataset_20x_he_norm.zip` contains only normalized H&E images and is not used here.

2. Extract the archive:

```bash
7z x ORIONCRC_dataset_tile_20x.zip
```

3. Convert the 17-channel mIF tiles into 16 paired mIHC targets and create deterministic dataset splits:

```bash
python tools/prepare_orion.py \
    --input-dir /path/to/ORIONCRC_dataset_tile_20x \
    --output-dir /path/to/Orion-CRC \
    --train-ratio 0.8 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --seed 42 \
    --workers 12
```

The split ratios can be changed as needed, but they must be positive and sum to `1.0`. The same seed always produces the same tile-level split, which is recorded in `split_manifest.csv`. Use `--max-samples` to convert a randomly sampled subset. The output directory must be new; pass `--overwrite` explicitly to replace an earlier generated dataset. Run `python tools/prepare_orion.py --help` for all options.

The script implements the mIF-driven synthesis described in our paper with the default parameters $\alpha=0.6$ and $\beta=0.8$. It excludes the PD-1 channel and produces the following HUSE-compatible structure:

```text
Orion-CRC/
├── train/
│   ├── he/
│   ├── CD3e/
│   ├── CD4/
│   └── ... (16 marker directories)
├── val/
├── test/
└── split_manifest.csv
```

The conversion creates 16 mIHC targets per H&E tile and therefore requires substantial additional disk space.

### Configuration

Each dataset has its own YAML config. The example provided here, [`configs/orion.yaml`](configs/orion.yaml), is set up for the **Orion-CRC** dataset (16 IHC markers, image size 256). To train on a different dataset, copy this file and adjust the values (e.g. `num_classes`, `data_root_*`, the marker list in `dataset_orion.py`), then point the launch script at your new YAML.

Fill in the dataset paths marked with `PATH/TO/...` before launching:

- `data_root_train` / `data_root_val`: dataset roots, organized as `<root>/he/<id>.<ext>` and `<root>/<MarkerName>/<id>.<ext>` with matching file names.
- `num_classes`: number of biomarkers (16 for Orion-CRC).
- `clip_anchor_dir`: directory containing the released CLIP anchor features used to initialize the Hi-MoE prototypes. It defaults to `weights/expert_anchors`.
- `init_moe_prototypes`: whether to initialize the three expert prototypes from the released anchors. It is enabled by default.
- `output_dir`: where checkpoints and logs are written.

Learning-rate schedule: warmup for `warmup_epochs`, hold `lr` until `decay_start_epoch`, then cosine-decay from `lr` to `min_lr` over `[decay_start_epoch, epochs]`. With the default `epochs=400`, `decay_start_epoch=250`, the LR stays at `1e-4` until epoch 250 and then decays to `1e-5` by epoch 400.

Any YAML value can be overridden on the command line (CLI flags take priority).

### Hi-MoE Expert Anchors

The repository includes three CLIP ViT-B/32 anchor features:

```text
weights/expert_anchors/
├── anchor_nuclear.pt
├── anchor_cytoplasm.pt
└── anchor_background.pt
```

Each file stores a 512-dimensional semantic feature for one histocomponent expert. At the start of training, HUSE projects these features into the model hidden space and uses them to initialize the nuclear, cytoplasm, and background expert prototypes in every Hi-MoE block. To train with random prototype initialization for an ablation, set `init_moe_prototypes: false` in the YAML config.

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

## 4. Acknowledgements

We thank the authors of [MIPHEI-ViT](https://github.com/sanofi-public/miphei-vit) for curating and publicly releasing the processed, paired ORION-CRC H&E-mIF tiles used by this project. We also acknowledge Lin et al. for the original [ORION-CRC dataset](https://doi.org/10.5281/zenodo.7637988). When using these data, please follow the license terms and citation requirements provided by the [MIPHEI-ViT dataset record](https://doi.org/10.5281/zenodo.15340874) and the original ORION-CRC release.

## 5. Citation

If you find this work useful in your research or use this code, please cite our paper:

```bibtex
@inproceedings{cen2026histocomponent,
  title={Histocomponent-Driven Universal Model for Virtual Immunohistochemistry Multiplex Staining via Joint Manifold Evolution},
  author={Cen, Jiajun and Xu, Siyuan and Gao, Lili and Wang, Yan},
  booktitle={European Conference on Computer Vision},
  pages={615--631},
  year={2026},
  organization={Springer}
}
```
