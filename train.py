#!/usr/bin/env python3
"""DeiT-S with a distillation token on ImageNet-1k, one node, DDP.

Launch (8 GPUs, global batch 1024):

    torchrun --standalone --nproc_per_node=8 train.py --data "$LOCAL/imagenet" --grad-accum-steps 1

Validation is sharded with DistributedSampler. Rank 0 writes checkpoints.
The student is wrapped in DDP; the frozen RegNetY-16GF teacher is one copy per GPU.
"""

import argparse
import math
import os
import random
import re
import time
from contextlib import nullcontext
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# --------------------------------------------------------------------------- dist


def init_distributed(backend):
    if 'RANK' not in os.environ or 'LOCAL_RANK' not in os.environ:
        raise SystemExit(
            'Launch with torchrun, for example:\n'
            '  torchrun --standalone --nproc_per_node=4 train.py --data "$LOCAL/imagenet"'
        )
    local_rank = int(os.environ['LOCAL_RANK'])
    # Long timeout: rank 0 validates the full split while the others sit on a barrier.
    dist.init_process_group(backend=backend, timeout=timedelta(hours=2))
    rank = dist.get_rank()
    world = dist.get_world_size()
    if backend == 'nccl':
        if not torch.cuda.is_available():
            raise SystemExit('NCCL requires CUDA. On a CPU self-check, use --self-check.')
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    return local_rank, rank, world, device


def unwrap(model):
    if isinstance(model, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        return model.module
    return model


def log(rank, *args):
    if rank == 0:
        print(*args, flush=True)


def broadcast_stop(stop, device):
    """Rank 0's stop decision. Every rank must call this or the job hangs."""
    flag = torch.zeros(1, device=device)
    if dist.get_rank() == 0 and stop:
        flag[0] = 1
    dist.broadcast(flag, src=0)
    return bool(flag.item() > 0.5)


def atomic_torch_save(obj, path):
    path = str(path)
    tmp = path + '.tmp'
    torch.save(obj, tmp)
    os.replace(tmp, path)


def cpu_count_for_job():
    # os.cpu_count() is the whole node under Slurm. Affinity is the allocation.
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 4


# --------------------------------------------------------------------------- data


class RASampler(torch.utils.data.Sampler):
    """Distributed repeated-augmentation sampler (facebookresearch/deit samplers.py).

    The generator seed is ``seed + epoch`` and does NOT include the rank. Every rank
    must build the same permutation and then stride it; adding the rank would
    give each GPU a different shuffle and overlapping or missing samples.
    """

    def __init__(self, dataset, num_replicas, rank, num_repeats=3, shuffle=True, seed=0):
        if num_repeats < 1:
            raise ValueError('num_repeats should be greater than 0')
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.num_repeats = num_repeats
        self.epoch = 0
        self.seed = seed
        self.shuffle = shuffle
        self.num_samples = int(math.ceil(len(dataset) * num_repeats / num_replicas))
        self.total_size = self.num_samples * num_replicas
        # Truncate to a multiple of 256, then split across ranks (DeiT).
        self.num_selected_samples = int(math.floor((len(dataset) // 256 * 256) / num_replicas))
        if self.num_selected_samples < 1:
            raise ValueError(f'RASampler selected 0 samples from len={len(dataset)} world={num_replicas}')

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)
        else:
            indices = torch.arange(len(self.dataset))
        indices = torch.repeat_interleave(indices, repeats=self.num_repeats, dim=0).tolist()
        padding = self.total_size - len(indices)
        if padding > 0:
            indices += indices[:padding]
        if len(indices) != self.total_size:
            raise RuntimeError('RASampler padding failed')
        indices = indices[self.rank:self.total_size:self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError('RASampler stride failed')
        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples


def make_transforms(image_size, val_resize, train):
    if train:
        from timm.data import create_transform
        return create_transform(
            input_size=image_size,
            is_training=True,
            scale=(0.08, 1.0),
            auto_augment='rand-m9-mstd0.5-inc1',
            interpolation='bicubic',
            re_prob=0.25,
            re_mode='pixel',
            re_count=1,
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        )
    return T.Compose([
        T.Resize(val_resize, interpolation=InterpolationMode.BICUBIC),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- loss


def require_distill_token(distillation_type, distill_token):
    if distillation_type in ('hard', 'soft') and not distill_token:
        raise ValueError(
            "distillation_type '%s' requires the distillation token. "
            "Official DeiT refuses to reuse the class head for the teacher loss "
            "(losses.py: outputs_kd is None)." % distillation_type
        )


class DistillationLoss(nn.Module):
    """deit/losses.py. Hard is Eq. 3; soft divides by numel, matching the paper code."""

    def __init__(self, base_criterion, teacher_model, distillation_type, alpha, tau):
        super().__init__()
        if distillation_type not in ('none', 'soft', 'hard'):
            raise ValueError(distillation_type)
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, labels):
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            zero = torch.zeros((), device=base_loss.device)
            return base_loss, base_loss.detach(), zero
        if outputs_kd is None:
            raise ValueError(
                'When knowledge distillation is enabled, the model is expected to return '
                'a Tuple[Tensor, Tensor] with the output of the class_token and the dist_token'
            )
        with torch.no_grad():
            teacher_outputs = self.teacher_model(inputs)
        if self.distillation_type == 'soft':
            temperature = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / temperature, dim=1),
                F.log_softmax(teacher_outputs / temperature, dim=1),
                reduction='sum',
                log_target=True,
            ) * (temperature * temperature) / outputs_kd.numel()
        else:
            distillation_loss = F.cross_entropy(outputs_kd, teacher_outputs.argmax(dim=1))
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss, base_loss.detach(), distillation_loss.detach()


def fuse_logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return (outputs[0] + outputs[1]) / 2
    return outputs


def topk_accuracy(logits, targets, topk=(1, 5)):
    if targets.ndim == 2:
        targets = targets.argmax(dim=1)
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    correct = pred.t().eq(targets.view(1, -1))
    out = []
    for k in topk:
        k = min(k, maxk)
        out.append(correct[:k].reshape(-1).float().sum(0) * 100.0 / targets.size(0))
    return out


# --------------------------------------------------------------------------- train


def train_one_epoch(model, loader, optimizer, scaler, device, criterion, mixup_fn,
                    ema, accum_steps, epoch, rank, limit_batches, amp):
    model.train()
    n_batches = len(loader) if not limit_batches else min(limit_batches, len(loader))
    if n_batches < 1:
        raise RuntimeError('empty training loader')
    optimizer.zero_grad(set_to_none=True)

    loss_sum = torch.zeros((), device=device)
    ce_sum = torch.zeros((), device=device)
    kd_sum = torch.zeros((), device=device)
    top1_sum = torch.zeros((), device=device)
    count = torch.zeros((), device=device)
    window_ok = torch.ones((), device=device)
    amp_on = amp and device.type == 'cuda'
    steps = 0

    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        hard = labels
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        # All-reduce once per optimizer step, not once per micro-batch.
        is_step = ((i + 1) % accum_steps == 0) or (i + 1 == n_batches)
        sync = nullcontext() if is_step else model.no_sync()
        with sync:
            with torch.amp.autocast('cuda', enabled=amp_on):
                outputs = model(images)
                loss, ce, kd = criterion(images, outputs, labels)
            scaler.scale(loss / accum_steps).backward()

        window_ok = torch.minimum(window_ok, torch.isfinite(loss.detach()).to(window_ok.dtype))
        bs = images.shape[0]
        loss_sum += loss.detach().float() * bs
        ce_sum += ce.float() * bs
        kd_sum += kd.float() * bs
        acc1, _ = topk_accuracy(fuse_logits(outputs).detach().float(), hard, topk=(1, 5))
        top1_sum += acc1.detach().float() * bs
        count += bs

        if is_step:
            dist.all_reduce(window_ok, op=dist.ReduceOp.MIN)
            if window_ok.item() == 0:
                raise RuntimeError(f'non-finite loss at epoch {epoch} batch {i}')
            window_ok = torch.ones((), device=device)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            steps += 1

        if rank == 0 and (i % 50 == 0 or i + 1 == n_batches):
            print(f'  epoch {epoch + 1}  {i + 1}/{n_batches}  '
                  f'loss {loss.detach().float().item():.4f}', flush=True)

    packed = torch.stack([loss_sum, ce_sum, kd_sum, top1_sum, count])
    dist.all_reduce(packed)
    denom = packed[4].clamp(min=1)
    return {
        'loss': (packed[0] / denom).item(),
        'ce': (packed[1] / denom).item(),
        'kd': (packed[2] / denom).item(),
        'top1': (packed[3] / denom).item(),
        'batches': n_batches,
        'steps': steps,
    }


@torch.no_grad()
def validate(model, loader, device, criterion, amp, limit_batches=0):
    model.eval()
    loss_sum = top1_sum = top5_sum = 0.0
    count = 0
    amp_on = amp and device.type == 'cuda'
    for i, (images, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp_on):
            logits = fuse_logits(model(images)).float()
        loss = criterion(logits, labels)
        acc1, acc5 = topk_accuracy(logits, labels, topk=(1, 5))
        bs = images.shape[0]
        loss_sum += loss.item() * bs
        top1_sum += acc1.item() * bs
        top5_sum += acc5.item() * bs
        count += bs
    if count == 0:
        return {'loss': float('nan'), 'top1': float('nan'), 'top5': float('nan')}
    return {'loss': loss_sum / count, 'top1': top1_sum / count, 'top5': top5_sum / count}


@torch.no_grad()
def validate_dist(model, loader, device, amp, limit_batches=0):
    """Every rank evaluates its own shard, then all_reduce the sums.

    50,000 ImageNet val images divide evenly by 2, 4 and 8, so drop_last=False
    does not pad. limit_batches is only for --smoke; every rank must use the same value.
    """
    model.eval()
    s = torch.zeros(4, device=device)  # loss, top1, top5, count
    for i, (images, labels) in enumerate(loader):
        if limit_batches and i >= limit_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=amp and device.type == 'cuda'):
            logits = fuse_logits(model(images)).float()
        a1, a5 = topk_accuracy(logits, labels, (1, 5))
        bs = images.shape[0]
        s += torch.stack([
            F.cross_entropy(logits, labels) * bs,
            a1 * bs,
            a5 * bs,
            torch.tensor(float(bs), device=device),
        ])
    dist.all_reduce(s)
    return {'loss': (s[0] / s[3]).item(), 'top1': (s[1] / s[3]).item(), 'top5': (s[2] / s[3]).item()}


class ModelEma:
    def __init__(self, model, decay):
        import copy
        self.module = copy.deepcopy(unwrap(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.updates = 0

    @torch.no_grad()
    def update(self, model):
        self.updates += 1
        d = min(self.decay, (1 + self.updates) / (10 + self.updates))
        src_sd = unwrap(model).state_dict()
        for k, v in self.module.state_dict().items():
            src = src_sd[k].detach()
            if v.dtype.is_floating_point:
                v.mul_(d).add_(src.to(v.device), alpha=1.0 - d)
            else:
                v.copy_(src.to(v.device))

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)


def gpu_recommendation(wall_epoch_seconds, world, n_epochs, targets):
    """GPUs ≈ (single_gpu_epoch_hours × epochs) / (target_hours × 0.9)."""
    single_hours = (wall_epoch_seconds / 3600.0) * world
    lines = [f'single-GPU epoch estimate {single_hours:.2f} h (linear scaling from {world} GPUs)']
    for target in targets:
        need = (single_hours * n_epochs) / (target * 0.9)
        on_this = (wall_epoch_seconds / 3600.0) * n_epochs
        lines.append(
            f'  {n_epochs} epochs on these {world} GPUs ≈ {on_this:.1f} h; '
            f'for a {target:.0f} h budget the formula wants about {need:.1f} GPUs'
        )
    return '\n'.join(lines)


# --------------------------------------------------------------------------- build


def build_student(args, device):
    import timm
    name = 'deit_small_distilled_patch16_224' if args.distill_token else 'deit_small_patch16_224'
    model = timm.create_model(
        name, pretrained=False, num_classes=1000,
        img_size=args.image_size, drop_path_rate=args.drop_path,
    )
    if args.distill_token:
        model.set_distilled_training(True)
    return model.to(device), name


def build_teacher(args, device, rank):
    import timm
    # Rank 0 downloads; the others wait so they don't corrupt the cache.
    if rank == 0:
        teacher = timm.create_model(args.teacher, pretrained=True, num_classes=1000)
    dist.barrier()
    if rank != 0:
        teacher = timm.create_model(args.teacher, pretrained=True, num_classes=1000)
    teacher = teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


def param_groups(model, weight_decay):
    skip = set()
    base = unwrap(model)
    if hasattr(base, 'no_weight_decay'):
        skip = set(base.no_weight_decay())
    decay, no_decay = [], []
    for name, p in base.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith('.bias') or any(k in name for k in skip):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ], len(decay), len(no_decay)


_WNID = re.compile(r'^n\d{8}$')


def imagenet1k_synsets():
    """ImageNet-1k classifier order as WordNet ids (n01440764), not display names.

    Current timm returns those ids from ``label_names()``. Other releases return
    lemmas such as ``tench, Tinca tinca``. ImageFolder.classes are the directory
    names, which on the PSC community tars are the ids. Compare against the
    synset list either way.
    """
    from timm.data import ImageNetInfo
    info = ImageNetInfo('imagenet-1k')
    if getattr(info, '_synsets', None):
        synsets = [str(s) for s in info._synsets]
    else:
        names = [str(n) for n in info.label_names()]
        if names and _WNID.match(names[0]):
            synsets = names
        else:
            import pkgutil
            raw = pkgutil.get_data('timm.data', '_info/imagenet_synsets.txt')
            if not raw:
                raise RuntimeError(
                    'timm label_names() are not WordNet ids and imagenet_synsets.txt '
                    'is not installed with timm, so the class-order check cannot run.'
                )
            synsets = raw.decode('utf-8').splitlines()
    synsets = [s.strip() for s in synsets if s.strip()]
    if len(synsets) != 1000 or any(not _WNID.match(s) for s in synsets):
        raise RuntimeError('ImageNet-1k synset list from timm is not 1000 WordNet ids.')
    return synsets


def check_class_order(classes, rank):
    synsets = imagenet1k_synsets()
    folders = [str(c) for c in classes]
    bad = [c for c in folders if not _WNID.match(c)]
    if bad:
        raise RuntimeError(
            'ImageNet class folders must be WordNet ids like n01440764, not display names. '
            f'First mismatches: {bad[:5]}'
        )
    if folders == synsets:
        log(rank, 'Class order matches ImageNet-1k synsets (n01440764, ...).')
        return
    folder_set = set(folders)
    synset_set = set(synsets)
    missing = [s for s in synsets if s not in folder_set]
    extra = [c for c in folders if c not in synset_set]
    if not missing and not extra:
        raise RuntimeError(
            'Class folders are the ImageNet-1k WordNet ids, but not in classifier index order. '
            f'folder[0]={folders[0]!r}, synset[0]={synsets[0]!r}. '
            'The teacher would supervise the wrong class.'
        )
    raise RuntimeError(
        'train/ does not match the ImageNet-1k synset list. '
        f'missing {len(missing)} (e.g. {missing[:3]}), extra {len(extra)} (e.g. {extra[:3]}).'
    )


# --------------------------------------------------------------------------- main train


def run_train(args):
    run_start = time.time()  # before dataset indexing; that scan is part of the budget
    local_rank, rank, world, device = init_distributed('nccl')
    require_distill_token(args.distillation_type, args.distill_token)

    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    if args.smoke:
        args.epochs = 2
        args.warmup_epochs = 1
        args.ema = False
        if args.limit_batches <= 0:
            args.limit_batches = 20

    eff_batch = args.batch_size * world * args.grad_accum_steps
    lr = args.base_lr * eff_batch / args.base_batch
    workers = args.workers
    if workers < 0:
        workers = max(1, cpu_count_for_job() // world)

    log(rank, f'world {world}  local_rank {local_rank}  device {device}')
    log(rank, f'batch/gpu {args.batch_size}  accum {args.grad_accum_steps}  '
              f'global batch {eff_batch}  lr {lr:.3e}')
    log(rank, f'epochs {args.epochs}  warmup {args.warmup_epochs}  '
              f'ema {args.ema_decay if args.ema else "off"}  '
              f'max hours {args.max_runtime_hours}')
    log(rank, f'workers/rank {workers}  (affinity {cpu_count_for_job()} cpus / {world} gpus)')
    if args.repeated_aug:
        log(rank, 'repeated aug ON (3x). 100 epochs sees about 33 distinct passes, '
                  'which is the paper setting. --no-repeated-aug makes it about 100 passes.')
    else:
        log(rank, 'repeated aug OFF. 100 epochs sees about 100 distinct passes.')
    log(rank, f'distill {args.distillation_type}  token {args.distill_token}  teacher {args.teacher}')

    os.makedirs(args.work_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.work_dir, 'checkpoints')
    if rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)
    dist.barrier()

    val_resize = int(round(args.image_size / 0.875))
    train_tf = make_transforms(args.image_size, val_resize, train=True)
    val_tf = make_transforms(args.image_size, val_resize, train=False)
    log(rank, 'Indexing ImageNet...')
    train_set = ImageFolder(os.path.join(args.data, 'train'), transform=train_tf)
    val_set = ImageFolder(os.path.join(args.data, 'val'), transform=val_tf)
    if train_set.classes != val_set.classes:
        raise RuntimeError('train and val class folders differ')
    if len(train_set.classes) != 1000:
        raise RuntimeError(f'expected 1000 classes, found {len(train_set.classes)}')
    check_class_order(train_set.classes, rank)

    if args.repeated_aug:
        train_sampler = RASampler(
            train_set, num_replicas=world, rank=rank, num_repeats=3, seed=args.seed,
        )
    else:
        train_sampler = DistributedSampler(
            train_set, num_replicas=world, rank=rank, shuffle=True, drop_last=True,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=seed_worker,
    )
    val_sampler = DistributedSampler(
        val_set, num_replicas=world, rank=rank, shuffle=False, drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size * 2,
        sampler=val_sampler,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
        worker_init_fn=seed_worker,
    )
    log(rank, f'train images {len(train_set):,}  batches/rank {len(train_loader):,}  '
              f'val images {len(val_set):,}  val batches/rank {len(val_loader):,}')

    from timm.data import Mixup
    mixup_fn = Mixup(
        mixup_alpha=0.8, cutmix_alpha=1.0, prob=1.0, switch_prob=0.5,
        mode='batch', label_smoothing=0.1, num_classes=1000,
    )

    model, student_name = build_student(args, device)
    n_params = sum(p.numel() for p in model.parameters())
    log(rank, f'student {student_name}  {n_params / 1e6:.2f}M')
    ema = ModelEma(model, args.ema_decay) if args.ema else None
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])

    teacher = None
    if args.distillation_type != 'none':
        teacher = build_teacher(args, device, rank)
        log(rank, f'teacher ready  {sum(p.numel() for p in teacher.parameters()) / 1e6:.1f}M frozen')

    from timm.loss import SoftTargetCrossEntropy
    from timm.scheduler import create_scheduler_v2
    alpha = 0.5 if args.distillation_type == 'hard' else (0.1 if args.distillation_type == 'soft' else 0.0)
    tau = 1.0 if args.distillation_type != 'soft' else 3.0
    criterion = DistillationLoss(
        SoftTargetCrossEntropy(), teacher, args.distillation_type, alpha, tau,
    )
    val_criterion = nn.CrossEntropyLoss()
    groups, n_decay, n_nodecay = param_groups(model, 0.05)
    optimizer = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.999), eps=1e-8)
    log(rank, f'AdamW decay tensors {n_decay}  no-decay {n_nodecay}')
    scheduler, _ = create_scheduler_v2(
        optimizer, sched='cosine', num_epochs=args.epochs,
        min_lr=1e-5, warmup_lr=1e-6, warmup_epochs=args.warmup_epochs, cooldown_epochs=0,
    )
    scaler = torch.amp.GradScaler('cuda', enabled=args.amp)

    start_epoch = 0
    best_top1 = 0.0
    history = []
    last_path = os.path.join(ckpt_dir, 'last.pth')
    best_path = os.path.join(ckpt_dir, 'best.pth')
    resume_path = args.resume or (None if args.fresh else (last_path if os.path.exists(last_path) else None))
    if resume_path:
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        unwrap(model).load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if ema is not None and ckpt.get('ema'):
            ema.load_state_dict(ckpt['ema'])
        if ckpt.get('scaler'):
            scaler.load_state_dict(ckpt['scaler'])
        start_epoch = int(ckpt['epoch']) + 1
        best_top1 = float(ckpt.get('best_top1', 0.0))
        history = list(ckpt.get('history', []))
        # Do not load the scheduler: its saved t_initial would pin the LR if epochs change.
        scheduler.step(start_epoch)
        log(rank, f'resumed {resume_path} at epoch {start_epoch}  best {best_top1:.2f}%  '
                  f'lr {optimizer.param_groups[0]["lr"]:.3e}')
    else:
        log(rank, 'training from scratch')

    run = None
    if rank == 0 and os.environ.get('WANDB_API_KEY'):
        import wandb
        run = wandb.init(
            project=os.environ.get('WANDB_PROJECT', 'imagenet1k-deit-s-distill'),
            name=os.environ.get('WANDB_NAME', 'deit_s_distill_in1k_psc'),
            config={
                'lr': lr, 'eff_batch': eff_batch, 'world': world,
                'epochs': args.epochs, 'distillation_type': args.distillation_type,
                'distill_token': args.distill_token, 'repeated_aug': args.repeated_aug,
                'ema_decay': args.ema_decay,
            },
        )

    if args.distillation_type != 'none' and not args.smoke and not args.benchmark_iters:
        probe = torch.zeros(2, device=device)
        if rank == 0:
            # Rank 0 only, on its own shard. The teacher is not DDP, so this is not collective.
            stats = validate(teacher, val_loader, device, val_criterion, args.amp, limit_batches=20)
            probe[0], probe[1] = stats['top1'], stats['top5']
            log(rank, f'teacher probe top-1 {stats["top1"]:.2f}%  top-5 {stats["top5"]:.2f}%')
        dist.broadcast(probe, src=0)
        if probe[0].item() < 50:
            raise RuntimeError(
                f'teacher probe top-1 is {probe[0].item():.1f}%. '
                'Class folders are probably not official ImageNet WNIDs.'
            )

    if args.benchmark_iters:
        warmup = min(10, args.benchmark_iters)
        log(rank, f'benchmark warmup {warmup} batches, then {args.benchmark_iters} timed')
        train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion, mixup_fn,
            ema, args.grad_accum_steps, 0, rank, warmup, args.amp,
        )
        dist.barrier()
        torch.cuda.synchronize()
        t0 = time.time()
        stats = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion, mixup_fn,
            ema, args.grad_accum_steps, 0, rank, args.benchmark_iters, args.amp,
        )
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        wall_epoch = elapsed * (len(train_loader) / max(stats['batches'], 1))
        log(rank, f'{stats["batches"]} batches in {elapsed:.1f}s  '
                  f'({elapsed / max(stats["batches"], 1):.3f} s/batch)  '
                  f'extrapolated epoch {wall_epoch / 3600:.2f} h')
        log(rank, gpu_recommendation(wall_epoch, world, args.epochs, (22, args.max_runtime_hours)))
        log(rank, 'benchmark does not write a checkpoint')
        dist.barrier()
        if run is not None:
            run.finish()
        dist.destroy_process_group()
        return

    last_epoch_time = None
    stop_reason = 'completed'
    for epoch in range(start_epoch, args.epochs):
        elapsed = time.time() - run_start
        should_stop = False
        if rank == 0 and last_epoch_time is not None:
            if elapsed + 1.2 * last_epoch_time > args.max_runtime_hours * 3600:
                should_stop = True
                log(rank, f'stopping before epoch {epoch + 1}: '
                          f'elapsed {elapsed / 3600:.2f}h + 1.2 x last epoch '
                          f'{last_epoch_time / 3600:.2f}h exceeds {args.max_runtime_hours}h')
        if broadcast_stop(should_stop, device):
            stop_reason = 'time budget'
            break

        train_sampler.set_epoch(epoch)
        epoch_start = time.time()
        curr_lr = optimizer.param_groups[0]['lr']
        log(rank, f'\n=== epoch {epoch + 1}/{args.epochs}  lr {curr_lr:.3e} ===')
        stats = train_one_epoch(
            model, train_loader, optimizer, scaler, device, criterion, mixup_fn,
            ema, args.grad_accum_steps, epoch, rank, args.limit_batches, args.amp,
        )
        log(rank, f'train  top1 {stats["top1"]:.2f}%  loss {stats["loss"]:.4f}  '
                  f'ce {stats["ce"]:.3f}  kd {stats["kd"]:.3f}')

        # Unwrapped module on every rank. DDP.forward is collective; the shard is not.
        val_limit = args.limit_batches if args.smoke else 0
        val_stats = validate_dist(unwrap(model), val_loader, device, args.amp, val_limit)
        ema_top1 = float('nan')
        if ema is not None:
            ema_stats = validate_dist(ema.module, val_loader, device, args.amp, val_limit)
            ema_top1 = ema_stats['top1']
        if rank == 0:
            log(rank, f'val    top1 {val_stats["top1"]:.2f}%  top5 {val_stats["top5"]:.2f}%  '
                      f'loss {val_stats["loss"]:.4f}')
            if ema is not None:
                log(rank, f'ema    top1 {ema_top1:.2f}%  top5 {ema_stats["top5"]:.2f}%')
            score = max(val_stats['top1'], ema_top1 if ema_top1 == ema_top1 else 0.0)
            row = {
                'epoch': epoch + 1, 'lr': curr_lr, 'train_loss': stats['loss'],
                'train_top1': stats['top1'], 'train_ce': stats['ce'], 'train_kd': stats['kd'],
                'val_top1': val_stats['top1'], 'val_top5': val_stats['top5'],
                'val_loss': val_stats['loss'], 'ema_top1': ema_top1,
            }
            history.append(row)
            payload = {
                'model': unwrap(model).state_dict(),
                'ema': ema.state_dict() if ema is not None else None,
                'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(),
                'epoch': epoch,
                'best_top1': max(best_top1, score),
                'history': history,
                'classes': train_set.classes,
                'args': vars(args),
            }
            atomic_torch_save(payload, last_path)
            if run is not None:
                import wandb
                run.log(row, step=epoch + 1)
                wandb.save(last_path)
            if score > best_top1:
                best_top1 = score
                atomic_torch_save(payload, best_path)
                if run is not None:
                    wandb.save(best_path)
                log(rank, f'new best {best_top1:.2f}%')
            if epoch == start_epoch and not args.limit_batches:
                # Includes this epoch's validation. The first epoch is often the slow one.
                log(rank, gpu_recommendation(
                    time.time() - epoch_start, world, args.epochs, (22, args.max_runtime_hours),
                ))
        dist.barrier()
        scheduler.step(epoch + 1)
        last_epoch_time = time.time() - epoch_start
        log(rank, f'epoch time {last_epoch_time / 60:.1f} min  '
                  f'elapsed {(time.time() - run_start) / 3600:.2f} h  best {best_top1:.2f}%')

    log(rank, f'done ({stop_reason}). best {best_top1:.2f}%  last {last_path}')
    if run is not None:
        run.finish()
    dist.barrier()
    dist.destroy_process_group()


# --------------------------------------------------------------------------- local checks


def check_local():
    """No GPUs and no torchrun: sampler, distillation guard, atomic save."""
    class _Len:
        def __init__(self, n):
            self.n = n
        def __len__(self):
            return self.n

    n, world = 4096, 4
    per_rank = []
    for r in range(world):
        sampler = RASampler(_Len(n), num_replicas=world, rank=r, num_repeats=3, seed=0)
        per_rank.append(list(sampler))
    lengths = {len(x) for x in per_rank}
    if lengths != {(n // 256 * 256) // world}:
        raise SystemExit(f'RASampler lengths {lengths}')
    again = list(RASampler(_Len(n), num_replicas=world, rank=0, num_repeats=3, seed=0))
    if again != per_rank[0]:
        raise SystemExit('RASampler is not deterministic for a fixed epoch')
    sampler = RASampler(_Len(n), num_replicas=world, rank=0, num_repeats=3, seed=0)
    sampler.set_epoch(1)
    if list(sampler) == per_rank[0]:
        raise SystemExit('set_epoch did not change the order')
    # ImageNet-1k train size, 4 ranks: one number, not a materialised index.
    imagenet_n = 1281167
    selected = (imagenet_n // 256 * 256) // 4
    if selected != 320256:
        raise SystemExit(f'unexpected ImageNet selected count {selected}')
    print(f'RASampler ok  imagenet selected/rank {selected}')

    try:
        require_distill_token('hard', False)
        raise SystemExit('hard distillation without a token should fail')
    except ValueError as exc:
        print('distill guard ok:', exc)

    class Teach(nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 10)

    loss_fn = DistillationLoss(nn.CrossEntropyLoss(), Teach(), 'hard', 0.5, 1.0)
    images = torch.zeros(4, 3, 8, 8)
    labels = torch.zeros(4, dtype=torch.long)
    try:
        loss_fn(images, torch.zeros(4, 10), labels)
        raise SystemExit('loss should reject a single head')
    except ValueError as exc:
        print('loss guard ok:', exc)
    value, ce, kd = loss_fn(images, (torch.zeros(4, 10), torch.zeros(4, 10)), labels)
    if not torch.isfinite(value):
        raise SystemExit('distillation loss is not finite')
    print(f'hard loss ok  {float(value):.4f}  ce {float(ce):.4f}  kd {float(kd):.4f}')

    path = '/tmp/deit_atomic_ckpt.pt'
    atomic_torch_save({'ok': True}, path)
    if not torch.load(path, weights_only=False)['ok'] or os.path.exists(path + '.tmp'):
        raise SystemExit('atomic save failed')
    os.remove(path)
    print('atomic save ok')
    synsets = imagenet1k_synsets()
    if synsets[0] != 'n01440764' or len(synsets) != 1000:
        raise SystemExit(f'unexpected synset list start {synsets[0]!r}')
    check_class_order(synsets, rank=0)
    try:
        check_class_order(['tench'] + synsets[1:], rank=0)
        raise SystemExit('display-name folders should fail the class-order check')
    except RuntimeError as exc:
        print('class-order guard ok:', exc)
    try:
        swapped = list(synsets)
        swapped[0], swapped[1] = swapped[1], swapped[0]
        check_class_order(swapped, rank=0)
        raise SystemExit('permuted WordNet ids should fail the class-order check')
    except RuntimeError as exc:
        print('class-order permutation guard ok:', exc)
    print('check-local passed')


class _FakeSet(Dataset):
    def __init__(self, n=1024, classes=10):
        self.n = n
        self.classes = classes
        g = torch.Generator().manual_seed(0)
        self.images = torch.randn(n, 3, 16, 16, generator=g)
        self.labels = torch.randint(0, classes, (n,), generator=g)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return self.images[idx], int(self.labels[idx])


class _TinyStudent(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.fc = nn.Linear(3 * 16 * 16, classes)
        self.fc_dist = nn.Linear(3 * 16 * 16, classes)

    def forward(self, x):
        flat = x.flatten(1)
        return self.fc(flat), self.fc_dist(flat)


class _TinyTeacher(nn.Module):
    def __init__(self, classes=10):
        super().__init__()
        self.fc = nn.Linear(3 * 16 * 16, classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


def run_self_check():
    """Two or more CPU ranks via torchrun. Exercises DDP, no_sync, barrier, stop, ckpt."""
    run_start = time.time()
    local_rank, rank, world, device = init_distributed('gloo')
    if device.type != 'cpu':
        raise RuntimeError('self-check is the gloo path')
    ds = _FakeSet()
    sampler = RASampler(ds, num_replicas=world, rank=rank, num_repeats=3, seed=0)
    loader = DataLoader(ds, batch_size=8, sampler=sampler, num_workers=0, drop_last=True)
    model = _TinyStudent()
    before = torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone()
    model = nn.parallel.DistributedDataParallel(model)
    teacher = _TinyTeacher().eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    criterion = DistillationLoss(nn.CrossEntropyLoss(), teacher, 'hard', 0.5, 1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
    scaler = torch.amp.GradScaler('cuda', enabled=False)

    # One short epoch, then the predictive stop must fire on every rank.
    sampler.set_epoch(0)
    stats = train_one_epoch(
        model, loader, optimizer, scaler, device, criterion, None,
        None, accum_steps=2, epoch=0, rank=rank, limit_batches=8, amp=False,
    )
    val_sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    val_loader = DataLoader(ds, batch_size=16, sampler=val_sampler, num_workers=0)
    val_stats = validate_dist(unwrap(model), val_loader, device, amp=False)
    if len(val_sampler) * world != len(ds):
        raise RuntimeError(f'val shard padded or dropped: {len(val_sampler)} x {world} != {len(ds)}')
    agreed = torch.tensor([val_stats['loss'], val_stats['top1'], val_stats['top5']], device=device)
    gathered_stats = [torch.zeros_like(agreed) for _ in range(world)]
    dist.all_gather(gathered_stats, agreed)
    if not all(torch.allclose(g, gathered_stats[0]) for g in gathered_stats):
        raise RuntimeError('sharded validation did not agree across ranks')
    if rank == 0:
        os.makedirs('/tmp/deit_self_check', exist_ok=True)
        atomic_torch_save({'epoch': 0, 'loss': stats['loss']}, '/tmp/deit_self_check/last.pth')
        print(f'self-check epoch loss {stats["loss"]:.4f}', flush=True)
    dist.barrier()
    last_epoch_time = time.time() - run_start
    # Budget already exceeded once a positive epoch time exists.
    should_stop = rank == 0 and (time.time() - run_start) + 1.2 * last_epoch_time > 0
    stopped = broadcast_stop(should_stop, device)
    if not stopped:
        raise RuntimeError('stop flag did not reach every rank')

    vec = torch.nn.utils.parameters_to_vector(unwrap(model).parameters()).detach()
    if torch.allclose(vec, before):
        raise RuntimeError('weights did not move')
    gathered = [torch.zeros_like(vec) for _ in range(world)]
    dist.all_gather(gathered, vec)
    for other in gathered[1:]:
        if not torch.allclose(other, gathered[0]):
            raise RuntimeError('ranks diverged')
    if rank == 0:
        ckpt = torch.load('/tmp/deit_self_check/last.pth', weights_only=False)
        if ckpt['epoch'] != 0 or not os.path.exists('/tmp/deit_self_check/last.pth'):
            raise RuntimeError('checkpoint missing')
        if os.path.exists('/tmp/deit_self_check/last.pth.tmp'):
            raise RuntimeError('tmp checkpoint left behind')
        print('self-check passed', flush=True)
    dist.barrier()
    dist.destroy_process_group()


def parse_args():
    p = argparse.ArgumentParser(description='DeiT-S distillation token, ImageNet-1k, DDP')
    p.add_argument('--data', default='', help='ImageNet root with train/ and val/')
    p.add_argument('--work-dir', default=os.path.expanduser('~/deit_s_in1k_distill'))
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--warmup-epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=128, help='per GPU')
    p.add_argument('--grad-accum-steps', type=int, default=2,
                   help='128 x 4 GPUs x 2 = 1024 global, lr 1e-3')
    p.add_argument('--base-lr', type=float, default=5e-4)
    p.add_argument('--base-batch', type=int, default=512)
    p.add_argument('--ema-decay', type=float, default=0.9999,
                   help='0.99996 would average over about 20 percent of a 100-epoch run')
    p.add_argument('--no-ema', action='store_true')
    p.add_argument('--max-runtime-hours', type=float, default=44.0)
    p.add_argument('--repeated-aug', action=argparse.BooleanOptionalAction, default=True,
                   help='3x RASampler, on by default (the paper setting). '
                        '--no-repeated-aug uses one pass per epoch.')
    p.add_argument('--distillation-type', default='hard', choices=('none', 'soft', 'hard'))
    p.add_argument('--distill-token', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--teacher', default='regnety_160.deit_in1k')
    p.add_argument('--image-size', type=int, default=224)
    p.add_argument('--drop-path', type=float, default=0.1)
    p.add_argument('--workers', type=int, default=-1, help='-1: affinity_cpus / world (~5 on this job)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', default='', help='checkpoint path; default is <work-dir>/checkpoints/last.pth')
    p.add_argument('--fresh', action='store_true', help='ignore an existing last.pth')
    p.add_argument('--smoke', action='store_true', help='2 epochs, 20 batches, no EMA')
    p.add_argument('--limit-batches', type=int, default=0)
    p.add_argument('--benchmark-iters', type=int, default=0,
                   help='time this many batches, print the GPU estimate, do not save')
    p.add_argument('--no-amp', action='store_true')
    p.add_argument('--check-local', action='store_true', help='sampler and loss checks, no torchrun')
    p.add_argument('--self-check', action='store_true',
                   help='CPU DDP smoke: torchrun --nproc_per_node=2 train.py --self-check')
    args = p.parse_args()
    args.ema = not args.no_ema
    args.amp = not args.no_amp
    if args.batch_size % 2 != 0:
        p.error('batch size must be even (timm Mixup)')
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.check_local:
        check_local()
    elif args.self_check:
        run_self_check()
    else:
        if not args.data:
            raise SystemExit('pass --data /path/to/imagenet (the folder that contains train/ and val/)')
        if not os.path.isdir(os.path.join(args.data, 'train')):
            raise SystemExit(f'missing {args.data}/train')
        if not os.path.isdir(os.path.join(args.data, 'val')):
            raise SystemExit(f'missing {args.data}/val')
        run_train(args)
