import os
import sys
import glob
import random
import time
import tempfile
import signal
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import box_iou
from tqdm import tqdm

def _find_data_root():
    """Pronadji dataset: prvo lokalno, pa kagglehub ako treba."""
    # Lokalne kandidat putanje
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_candidates = [
        os.path.join(script_dir, "Dataset"),
        os.path.join(script_dir, "pcb-defect-dataset"),
        "Dataset",
        "pcb-defect-dataset",
    ]
    for candidate in local_candidates:
        if os.path.isdir(candidate) and os.path.isdir(os.path.join(candidate, "train")):
            return candidate
    # Ako nema lokalno, probaj kagglehub
    try:
        import kagglehub
        path = kagglehub.dataset_download("norbertelter/pcb-defect-dataset")
        return os.path.join(path, "pcb-defect-dataset")
    except Exception:
        return os.path.join(script_dir, "pcb-defect-dataset")


DATA_ROOT = _find_data_root()
NUM_DEFECT_CLASSES = 6
NUM_EPOCHS = 12
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
NUM_WORKERS = 0
PRINT_FREQ = 20
CHECKPOINT_DIR = "checkpoints"



def save_checkpoint_atomic(state, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(filename))
    os.close(tmp_fd)
    torch.save(state, tmp_path)
    os.replace(tmp_path, filename)


def yolo_xywh_to_xyxy(xc, yc, w, h, img_w, img_h):
    cx = float(xc) * img_w
    cy = float(yc) * img_h
    bw = float(w) * img_w
    bh = float(h) * img_h
    x_min = max(0.0, cx - bw / 2.0)
    y_min = max(0.0, cy - bh / 2.0)
    x_max = min(float(img_w), cx + bw / 2.0)
    y_max = min(float(img_h), cy + bh / 2.0)
    return [x_min, y_min, x_max, y_max]


class PCBYoloDataset(Dataset):
    #Klasa koja vraca sliku sa xyxy labelama pretvorene u tensor
    def __init__(self, split_folder, transforms=None):
        self.img_dir = os.path.join(split_folder, "images")
        self.label_dir = os.path.join(split_folder, "labels")
        self.transforms = transforms
        self.images = sorted(glob.glob(os.path.join(self.img_dir, "*.*")))
        if not self.images:
            raise RuntimeError(f"No images found in {self.img_dir}")
        print(f"  {os.path.basename(split_folder)}: {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        base = Path(img_path).stem
        label_path = os.path.join(self.label_dir, base + ".txt")

        #Ako ne postoji label fajl, probaj sa nadji _256
        if not os.path.exists(label_path) and "_600" in base:
            alt_base = base.replace("_600", "_256")
            alt_path = os.path.join(self.label_dir, alt_base + ".txt")
            if os.path.exists(alt_path):
                label_path = alt_path

        boxes, labels = [], []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls = int(float(parts[0]))
                    xyxy = yolo_xywh_to_xyxy(
                        float(parts[1]), float(parts[2]),
                        float(parts[3]), float(parts[4]),
                        img_w, img_h
                    )
                    boxes.append(xyxy)
                    labels.append(cls + 1)

        if boxes:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros(len(labels), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes, "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": area, "iscrowd": iscrowd,
        }

        if self.transforms:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)

        return img, target


class TrainTransform:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob
        self.color_jitter = torchvision.transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.15, hue=0.05
        )

    def __call__(self, img, target):
        img = self.color_jitter(img)
        img = F.to_tensor(img)
        if random.random() < self.flip_prob and target["boxes"].shape[0] > 0:
            img = F.hflip(img)
            _, h, w = img.shape
            boxes = target["boxes"]
            x_min = w - boxes[:, 2]
            x_max = w - boxes[:, 0]
            boxes[:, 0] = x_min
            boxes[:, 2] = x_max
            target["boxes"] = boxes
        return img, target


def collate_fn(batch):
    return tuple(zip(*batch))


#mAP evaluacija
def _compute_ap(recalls, precisions):
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([1.0], precisions, [0.0]))
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for t in np.linspace(0.0, 1.0, 101):
        p = precisions[recalls >= t]
        ap += (p.max() if len(p) > 0 else 0.0)
    return ap / 101.0


def _evaluate_single_iou(all_dets, all_gts, iou_thresh, num_classes):
    aps = []
    for cls in range(1, num_classes):
        det_boxes, det_scores, det_img_ids = [], [], []
        gt_per_img, gt_matched = {}, {}
        n_gt = 0
        for img_id, (d, g) in enumerate(zip(all_dets, all_gts)):
            mask = d["labels"] == cls
            if mask.any():
                det_boxes.append(d["boxes"][mask])
                det_scores.append(d["scores"][mask])
                det_img_ids.extend([img_id] * mask.sum().item())
            gt_mask = g["labels"] == cls
            gt_b = g["boxes"][gt_mask]
            gt_per_img[img_id] = gt_b
            gt_matched[img_id] = torch.zeros(gt_b.shape[0], dtype=torch.bool)
            n_gt += gt_b.shape[0]

        if n_gt == 0:
            continue
        if not det_boxes:
            aps.append(0.0)
            continue

        det_boxes = torch.cat(det_boxes)
        det_scores = torch.cat(det_scores)
        det_img_ids = torch.tensor(det_img_ids, dtype=torch.long)
        order = det_scores.argsort(descending=True)

        tp = np.zeros(len(order))
        fp = np.zeros(len(order))
        for di, idx in enumerate(order):
            iid = det_img_ids[idx].item()
            db = det_boxes[idx].unsqueeze(0)
            gb = gt_per_img[iid]
            if gb.shape[0] == 0:
                fp[di] = 1
                continue
            ious = box_iou(db, gb)[0]
            best_iou, best_gt = ious.max(dim=0)
            if best_iou.item() >= iou_thresh and not gt_matched[iid][best_gt]:
                tp[di] = 1
                gt_matched[iid][best_gt] = True
            else:
                fp[di] = 1

        tp_c = np.cumsum(tp)
        fp_c = np.cumsum(fp)
        aps.append(_compute_ap(tp_c / n_gt, tp_c / (tp_c + fp_c)))
    return np.mean(aps) if aps else 0.0


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_dets, all_gts = [], []
    for images, targets in tqdm(loader, desc="Eval", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for o, t in zip(outputs, targets):
            all_dets.append({k: v.cpu() for k, v in o.items()})
            all_gts.append({"boxes": t["boxes"].cpu(), "labels": t["labels"].cpu()})
    nc = NUM_DEFECT_CLASSES + 1
    map50 = _evaluate_single_iou(all_dets, all_gts, 0.5, nc)
    maps = [_evaluate_single_iou(all_dets, all_gts, t, nc) for t in np.arange(0.5, 1.0, 0.05)]
    model.train()
    return {"mAP@0.5": map50, "mAP@0.5:0.95": np.mean(maps)}

def get_model(num_classes):
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def main():
    num_classes = NUM_DEFECT_CLASSES + 1
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    _interrupt = {"value": False}

    def _sigint_handler(signum, frame):
        if _interrupt["value"]:
            sys.exit(1)
        _interrupt["value"] = True

    signal.signal(signal.SIGINT, _sigint_handler)

    print(f"Device: {DEVICE}")
    print(f"Data root: {DATA_ROOT}")
    print("Loading datasets")

    train_ds = PCBYoloDataset(os.path.join(DATA_ROOT, "train"), transforms=TrainTransform())
    val_ds = PCBYoloDataset(os.path.join(DATA_ROOT, "val"))
    test_ds = PCBYoloDataset(os.path.join(DATA_ROOT, "test"))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False,
                             num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    model = get_model(num_classes).to(DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, weight_decay=1e-4)

    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=500)
    decay = torch.optim.lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.1)
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup, decay], milestones=[1])

    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

    best_map = 0.0
    global_step = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")

        for i, (images, targets) in enumerate(pbar):
            images = [img.to(DEVICE) for img in images]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += losses.item()
            global_step += 1

            if (i + 1) % PRINT_FREQ == 0:
                avg = running_loss / PRINT_FREQ
                pbar.set_postfix(loss=f"{avg:.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")
                running_loss = 0.0

            if _interrupt["value"]:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"interrupt_epoch{epoch}_step{global_step}.pth")
                torch.save({
                    "epoch": epoch, "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, ckpt_path)
                print(f"\nSaved interrupt checkpoint: {ckpt_path}")
                sys.exit(0)

        scheduler.step()
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch} done in {elapsed:.1f}s")

        # Evaluacija
        metrics = evaluate(model, val_loader, DEVICE)
        print(f"  Val mAP@0.5: {metrics['mAP@0.5']:.4f}  mAP@0.5:0.95: {metrics['mAP@0.5:0.95']:.4f}")

        ckpt_path = os.path.join(CHECKPOINT_DIR, f"fasterrcnn_pcb_epoch{epoch}.pth")
        torch.save({
            "epoch": epoch, "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_mAP": metrics["mAP@0.5"],
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # Best model
        if metrics["mAP@0.5"] > best_map:
            best_map = metrics["mAP@0.5"]
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best mAP@0.5: {best_map:.4f}")

    # Finalna evaluacija
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_model.pth"), weights_only=True))
    test_metrics = evaluate(model, test_loader, DEVICE)
    print(f"Test mAP@0.5:      {test_metrics['mAP@0.5']:.4f}")
    print(f"Test mAP@0.5:0.95: {test_metrics['mAP@0.5:0.95']:.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
