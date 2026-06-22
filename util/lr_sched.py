import math


# def adjust_learning_rate(optimizer, epoch, args):
#     """Decay the learning rate with half-cycle cosine after warmup"""
#     if epoch < args.warmup_epochs:
#         lr = args.lr * epoch / args.warmup_epochs 
#     else:
#         if args.lr_schedule == "constant":
#             lr = args.lr
#         elif args.lr_schedule == "cosine":
#             lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
#                 (1. + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)))
#         else:
#             raise NotImplementedError
#     for param_group in optimizer.param_groups:
#         if "lr_scale" in param_group:
#             param_group["lr"] = lr * param_group["lr_scale"]
#         else:
#             param_group["lr"] = lr
#     return lr

def adjust_learning_rate(optimizer, epoch, args):
    """
    args.decay_start_epoch: 在此之前保持 args.lr，在此之后开始余弦退火
    """
    # 获取参数，如果没有传这就默认 250 (做个保底)
    decay_start = getattr(args, 'decay_start_epoch', 250)
    
    # 1. Warmup
    if args.warmup_epochs > 0 and epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs 
        
    # 2. 平台期 (Plateau): Warmup 后，保持 args.lr 直到 decay_start
    elif epoch < decay_start:
        lr = args.lr
        
    # 3. 余弦退火 (Cosine Decay)
    else:
        # 现在的进度是相对于 (总epoch - 开始decay的epoch)
        eff_epoch = epoch - decay_start
        eff_total_epochs = args.epochs - decay_start

        if eff_total_epochs <= 0:
            lr = args.min_lr
        else:
            progress = eff_epoch / eff_total_epochs
            progress = min(progress, 1.0)

            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * \
                 (1. + math.cos(math.pi * progress))

    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
            
    return lr
