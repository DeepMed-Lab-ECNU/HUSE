import argparse
import copy
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func
from torch.utils.data import DataLoader, Dataset
from torchmetrics.image.kid import KernelInceptionDistance
from tqdm import tqdm

from dataset_orion import MARKER_LIST
from model_jit import JiT_models


class MyDenoiser(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


class OrionEvalDataset(Dataset):
    def __init__(self, root, marker_name, img_size):
        self.img_size = img_size
        self.he_dir = os.path.join(root, "he")
        self.marker_dir = os.path.join(root, marker_name)
        if not os.path.exists(self.he_dir):
            raise FileNotFoundError(f"HE directory not found: {self.he_dir}")
        if not os.path.exists(self.marker_dir):
            raise FileNotFoundError(f"Marker directory not found: {self.marker_dir}")

        he_files = sorted([f for f in os.listdir(self.he_dir) if f.endswith((".jpg", ".png", ".tif"))])
        self.files = [f for f in he_files if os.path.exists(os.path.join(self.marker_dir, f))]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filename = self.files[index]
        he_img = Image.open(os.path.join(self.he_dir, filename)).convert("RGB")
        gt_img = Image.open(os.path.join(self.marker_dir, filename)).convert("RGB")
        he_img = TF.resize(he_img, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BICUBIC)
        gt_img = TF.resize(gt_img, (self.img_size, self.img_size), interpolation=TF.InterpolationMode.BICUBIC)
        he_tensor = TF.to_tensor(he_img) * 2.0 - 1.0
        gt_tensor = TF.to_tensor(gt_img) * 2.0 - 1.0
        return he_tensor, gt_tensor, filename


def compute_metrics(pred_np, gt_np):
    psnr = psnr_func(gt_np, pred_np, data_range=255)
    ssim = ssim_func(gt_np, pred_np, data_range=255, channel_axis=2)
    return psnr, ssim


def load_model(args, device):
    model_inner = JiT_models[args.model](
        input_size=args.img_size,
        num_classes=len(MARKER_LIST),
        in_channels=6,
        clip_dim=args.clip_dim,
        clip_anchor_dir=args.clip_anchor_dir,
    )
    model_container = MyDenoiser(model_inner)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = ckpt["ema_model"] if "ema_model" in ckpt else ckpt["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model_container.ema_model.load_state_dict(new_state_dict, strict=False)
    model = model_container.ema_model.to(device)
    model.eval()
    return model


@torch.no_grad()
def sample_batch(model, he_img, marker_idx, args, device):
    batch_size = he_img.shape[0]
    labels = torch.full((batch_size,), marker_idx, device=device, dtype=torch.long)
    curr_z = torch.randn_like(he_img) * args.noise_scale
    dt = 1.0 / args.num_sampling_steps

    for step in range(args.num_sampling_steps):
        t_val = step / args.num_sampling_steps
        t_curr = torch.full((batch_size,), t_val, device=device)
        model_input = torch.cat([he_img, curr_z], dim=1)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            ret = model(model_input, t_curr, labels)
            x_pred_raw = ret[0] if isinstance(ret, tuple) else ret
        x_pred = x_pred_raw[:, 3:, :, :]
        v_pred = (x_pred - curr_z) / max(1 - t_val, 1e-5)
        curr_z = curr_z + v_pred * dt

    return curr_z.clamp(-1, 1)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="HUSE Orion evaluation: save images + log metrics")
    parser.add_argument("--data_path", type=str, default="PATH/TO/Orion-CRC/test")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./orion_eval_outputs")
    parser.add_argument("--output_txt", type=str, default="./orion_eval_outputs/metrics.txt")
    parser.add_argument("--model", type=str, default="JiT-B/16")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--orig_size", type=int, default=333)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--noise_scale", type=float, default=2.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clip_dim", type=int, default=512)
    parser.add_argument("--clip_anchor_dir", type=str, default="PATH/TO/CLIP_ANCHOR_FEATURES")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)
    output_txt_dir = os.path.dirname(args.output_txt)
    if output_txt_dir:
        os.makedirs(output_txt_dir, exist_ok=True)
    for marker_name in MARKER_LIST:
        os.makedirs(os.path.join(args.output_dir, marker_name), exist_ok=True)

    print(f"Loading checkpoint from {args.checkpoint}")
    model = load_model(args, device)

    all_psnr = []
    all_ssim = []
    all_kid = []

    with open(args.output_txt, "w") as log_f:
        log_f.write("=== HUSE Orion Evaluation Report ===\n")
        log_f.write(f"Checkpoint: {args.checkpoint}\n")
        log_f.write(f"Steps: {args.num_sampling_steps} | Noise Scale: {args.noise_scale}\n")
        log_f.write(f"{'Marker':<12} | {'PSNR':<10} | {'SSIM':<10} | {'KID(x1000)':<12} | Count\n")
        log_f.write("-" * 72 + "\n")

        for marker_idx, marker_name in enumerate(MARKER_LIST):
            dataset = OrionEvalDataset(args.data_path, marker_name, args.img_size)
            if len(dataset) == 0:
                continue

            loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
            psnr_list = []
            ssim_list = []
            kid_metric = KernelInceptionDistance(
                subset_size=min(50, len(dataset)),
                normalize=False,
            ).to(device)

            print(f"Processing {marker_name} ({len(dataset)} samples)")
            for he_img, gt_img, filenames in tqdm(loader, desc=marker_name):
                he_img = he_img.to(device).float()
                gt_img = gt_img.to(device)
                pred_img = sample_batch(model, he_img, marker_idx, args, device)

                pred_uint8 = ((pred_img + 1) / 2.0 * 255.0).to(dtype=torch.uint8)
                gt_uint8 = ((gt_img.clamp(-1, 1) + 1) / 2.0 * 255.0).to(dtype=torch.uint8)
                kid_metric.update(gt_uint8, real=True)
                kid_metric.update(pred_uint8, real=False)

                for i, filename in enumerate(filenames):
                    pred_np = pred_uint8[i].cpu().numpy().transpose(1, 2, 0)
                    gt_np = gt_uint8[i].cpu().numpy().transpose(1, 2, 0)
                    psnr, ssim = compute_metrics(pred_np, gt_np)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    pred_pil = Image.fromarray(pred_np)
                    if args.orig_size != args.img_size:
                        pred_pil = pred_pil.resize((args.orig_size, args.orig_size), Image.BICUBIC)
                    pred_pil.save(os.path.join(args.output_dir, marker_name, filename))

            avg_psnr = float(np.mean(psnr_list))
            avg_ssim = float(np.mean(ssim_list))
            kid_value = kid_metric.compute()[0].item() * 1000

            all_psnr.append(avg_psnr)
            all_ssim.append(avg_ssim)
            all_kid.append(kid_value)

            result_line = (
                f"{marker_name:<12} | {avg_psnr:<10.4f} | {avg_ssim:<10.4f} | "
                f"{kid_value:<12.4f} | {len(dataset)}"
            )
            print(result_line)
            log_f.write(result_line + "\n")

        if all_psnr:
            log_f.write("-" * 72 + "\n")
            avg_line = (
                f"{'AVERAGE':<12} | {np.mean(all_psnr):<10.4f} | {np.mean(all_ssim):<10.4f} | "
                f"{np.mean(all_kid):<12.4f} | {len(all_psnr)} markers"
            )
            print(avg_line)
            log_f.write(avg_line + "\n")


if __name__ == "__main__":
    main()
