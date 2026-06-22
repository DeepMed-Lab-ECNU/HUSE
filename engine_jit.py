import math
import sys

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim_func
from skimage.metrics import peak_signal_noise_ratio as psnr_func

import util.misc as misc
import util.lr_sched as lr_sched


def compute_metrics(pred_img, gt_img):
    """pred_img / gt_img: numpy array, [H, W, 3], uint8 in [0, 255]."""
    psnr = psnr_func(gt_img, pred_img, data_range=255)
    ssim = ssim_func(gt_img, pred_img, data_range=255, channel_axis=2)
    return psnr, ssim


def train_one_epoch(model, model_without_ddp, data_loader, optimizer, device, epoch, args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 50

    optimizer.zero_grad()
    jit_model = model_without_ddp.model

    for data_iter_step, (he, ihc, labels) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        he = he.to(device, non_blocking=True).to(torch.float32)
        ihc = ihc.to(device, non_blocking=True).to(torch.float32)
        labels = labels.to(device, non_blocking=True)

        # logit-normal time sampling + flow-matching interpolation
        s = torch.randn(ihc.shape[0], device=device) * args.P_std + args.P_mean
        t = torch.sigmoid(s)
        eps = torch.randn_like(ihc) * args.noise_scale

        t_expand = t.view(-1, 1, 1, 1)
        z_t = t_expand * ihc + (1 - t_expand) * eps

        # Joint Manifold Anchoring: X_t = [H&E || z_t]
        model_input = torch.cat([he, z_t], dim=1)

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            x_pred_raw, _, all_gates = jit_model(model_input, t, labels)
            x_pred = x_pred_raw[:, 3:, :, :]            # last 3 channels = predicted IHC
            v_target = ihc - eps
            v_pred = (x_pred - z_t) / (1 - t_expand).clamp(min=1e-5)
            loss_V = torch.mean((v_pred - v_target) ** 2)
        loss = loss_V
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()

        model_without_ddp.update_ema()

        layer_means = [g.mean() for g in all_gates]
        avg_gate_value = torch.stack(layer_means).mean().item()

        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_V=loss_V.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(avg_gate=avg_gate_value)


@torch.no_grad()
def evaluate_vs(model_container, data_loader, device, epoch, args):
    eval_model = model_container.ema_model
    eval_model.eval()

    print(f"--- Start Evaluation at Epoch {epoch} ---")
    psnr_list = []
    ssim_list = []

    for he, gt_ihc, labels in data_loader:
        he = he.to(device).to(torch.float32)
        gt_ihc = gt_ihc.to(device).to(torch.float32)
        labels = labels.to(device)

        # start from random noise and integrate the velocity field (Euler solver)
        z = torch.randn_like(gt_ihc) * args.noise_scale
        dt = 1.0 / args.num_sampling_steps
        curr_z = z

        for step in range(args.num_sampling_steps):
            t_curr_val = step / args.num_sampling_steps
            t_curr = torch.full((he.shape[0],), t_curr_val, device=device)

            model_input = torch.cat([he, curr_z], dim=1)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                x_pred_raw, _, _ = eval_model(model_input, t_curr, labels)

            x_pred = x_pred_raw[:, 3:, :, :]
            denom = max(1 - t_curr_val, 1e-5)
            v_pred = (x_pred - curr_z) / denom
            curr_z = curr_z + v_pred * dt

        pred_img = (curr_z.clamp(-1, 1) + 1) / 2.0 * 255.0
        gt_img = (gt_ihc.clamp(-1, 1) + 1) / 2.0 * 255.0

        pred_np = pred_img.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)
        gt_np = gt_img.cpu().numpy().astype(np.uint8).transpose(0, 2, 3, 1)

        for b in range(pred_np.shape[0]):
            p, s = compute_metrics(pred_np[b], gt_np[b])
            psnr_list.append(p)
            ssim_list.append(s)

    avg_psnr = float(np.mean(psnr_list))
    avg_ssim = float(np.mean(ssim_list))
    print(f"Epoch {epoch} Eval Result: PSNR={avg_psnr:.4f}, SSIM={avg_ssim:.4f}")
    return avg_psnr, avg_ssim
