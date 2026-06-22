import argparse
import copy
import os
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import yaml

import util.misc as misc
from dataset_orion import OrionDataset, MARKER_LIST
from engine_jit import evaluate_vs, train_one_epoch
from model_jit import JiT_models


class MyDenoiser(torch.nn.Module):
    def __init__(self, model, ema_decay=0.999):
        super().__init__()
        self.model = model
        self.ema_model = copy.deepcopy(model)
        for p in self.ema_model.parameters():
            p.requires_grad_(False)
        self.ema_decay = ema_decay

    def update_ema(self):
        with torch.no_grad():
            for p, ema_p in zip(self.model.parameters(), self.ema_model.parameters()):
                ema_p.data = ema_p.data * self.ema_decay + p.data * (1 - self.ema_decay)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


def save_checkpoint(args, model_without_ddp, optimizer, epoch, filename):
    if misc.get_rank() == 0:
        output_dir = Path(args.output_dir)
        save_path = output_dir / filename
        checkpoint = {
            'model': model_without_ddp.model.state_dict(),
            'ema_model': model_without_ddp.ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'args': args,
        }
        torch.save(checkpoint, save_path)
        print(f"Checkpoint saved to {save_path}")


def get_args_parser():
    parser = argparse.ArgumentParser('HUSE Virtual IHC Staining', add_help=False)
    # config file
    parser.add_argument('--config', default='configs/orion.yaml', type=str,
                        help='Path to a YAML config file. CLI args override YAML values.')
    # model
    parser.add_argument('--model', default='JiT-B/16', type=str)
    parser.add_argument('--img_size', default=256, type=int)
    parser.add_argument('--num_classes', default=16, type=int)
    parser.add_argument('--in_channels', default=6, type=int)
    parser.add_argument('--clip_dim', default=512, type=int)
    parser.add_argument('--clip_anchor_dir', default='PATH/TO/CLIP_ANCHOR_FEATURES', type=str)
    parser.add_argument('--init_moe_prototypes', action='store_true',
                        help='Initialize Hi-MoE prototypes from CLIP anchor features.')
    # data
    parser.add_argument('--data_root_train', default='PATH/TO/Orion-CRC/train', type=str)
    parser.add_argument('--data_root_val', default='PATH/TO/Orion-CRC/val', type=str)
    # optimization
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--blr', type=float, default=5e-5)
    parser.add_argument('--min_lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--lr_schedule', default='constant', type=str, help='constant or cosine')
    parser.add_argument('--decay_start_epoch', default=250, type=int)
    parser.add_argument('--ema_decay', default=0.999, type=float)
    # diffusion / flow-matching
    parser.add_argument('--P_mean', default=-0.8, type=float)
    parser.add_argument('--P_std', default=0.8, type=float)
    parser.add_argument('--noise_scale', default=2.0, type=float)
    parser.add_argument('--num_sampling_steps', default=50, type=int)
    parser.add_argument('--proj_dropout', type=float, default=0.0)
    # io / logging
    parser.add_argument('--output_dir', default='./output_vs', type=str)
    parser.add_argument('--save_last_freq', default=50, type=int)
    parser.add_argument('--eval_freq', default=5, type=int)
    parser.add_argument('--resume', default='', type=str)
    # runtime
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true')
    # distributed
    parser.add_argument('--world_size', default=1, type=int)
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://')
    return parser


def load_config_into_args(parser, argv=None):
    """Parse CLI once to find --config, load YAML as defaults, then re-parse so that
    explicit CLI flags still override the YAML values."""
    args, _ = parser.parse_known_args(argv)
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f) or {}
        valid_keys = {a.dest for a in parser._actions}
        cfg = {k: v for k, v in cfg.items() if k in valid_keys}
        parser.set_defaults(**cfg)
        print(f"[config] Loaded defaults from {args.config}")
    return parser.parse_args(argv)


def main(args):
    misc.init_distributed_mode(args)
    print("Args:", args)

    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    if misc.get_rank() == 0 and args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    dataset_train = OrionDataset(mode='train', img_size=args.img_size, root=args.data_root_train)
    dataset_val = OrionDataset(mode='val', img_size=args.img_size, root=args.data_root_val)
    print(f"Train Size: {len(dataset_train)}, Val Size: {len(dataset_val)}")

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
    )
    sampler_val = torch.utils.data.DistributedSampler(
        dataset_val, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=False
    )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
    )

    model_inner = JiT_models[args.model](
        input_size=args.img_size,
        num_classes=args.num_classes,
        in_channels=args.in_channels,
        proj_drop=args.proj_dropout,
        clip_dim=args.clip_dim,
        clip_anchor_dir=args.clip_anchor_dir,
    )
    if args.init_moe_prototypes:
        model_inner.init_moe_prototypes()

    model = MyDenoiser(model_inner, ema_decay=args.ema_decay)
    model.to(device)
    print("Actual LR:", args.lr)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    param_groups = [{
        'params': [p for p in model_without_ddp.model.parameters() if p.requires_grad],
        'weight_decay': args.weight_decay,
        'lr': args.lr,
    }]
    optimizer = torch.optim.AdamW(param_groups)

    if args.resume:
        print(f"Loading checkpoint from {args.resume}...")
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        load_msg = model_without_ddp.model.load_state_dict(checkpoint['model'], strict=False)
        if misc.get_rank() == 0 and (load_msg.missing_keys or load_msg.unexpected_keys):
            print(f"[Resume:model] missing_keys={load_msg.missing_keys}, unexpected_keys={load_msg.unexpected_keys}")
        if 'ema_model' in checkpoint:
            ema_load_msg = model_without_ddp.ema_model.load_state_dict(checkpoint['ema_model'], strict=False)
            if misc.get_rank() == 0 and (ema_load_msg.missing_keys or ema_load_msg.unexpected_keys):
                print(f"[Resume:ema] missing_keys={ema_load_msg.missing_keys}, unexpected_keys={ema_load_msg.unexpected_keys}")
        else:
            print("Warning: No EMA model found in checkpoint.")
        if 'optimizer' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {checkpoint.get('epoch', 0)}")
    else:
        args.start_epoch = 0

    print(f"Start training for {args.epochs} epochs")
    best_psnr = 0.0
    best_ssim = 0.0

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(model, model_without_ddp, data_loader_train, optimizer, device, epoch, args)

        if (epoch % args.save_last_freq == 0 and epoch > 0) or (epoch + 1 == args.epochs):
            save_checkpoint(args, model_without_ddp, optimizer, epoch, "checkpoint-last.pth")

        if (epoch % args.eval_freq == 0 and epoch > 0) or (epoch + 1 == args.epochs):
            val_psnr, val_ssim = evaluate_vs(model_without_ddp, data_loader_val, device, epoch, args)
            if val_psnr > best_psnr and val_ssim > best_ssim:
                print(f"*** New Best! Epoch {epoch} | PSNR: {val_psnr:.4f} | SSIM: {val_ssim:.4f}")
                best_psnr = val_psnr
                best_ssim = val_ssim
                save_checkpoint(args, model_without_ddp, optimizer, epoch, "checkpoint-best.pth")


if __name__ == '__main__':
    parser = get_args_parser()
    args = load_config_into_args(parser)
    main(args)
