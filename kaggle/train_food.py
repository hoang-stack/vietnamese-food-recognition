"""30VNFoods on a Kaggle T4: data audit, four models, calibration data and an ONNX export.

Writes only small files to /kaggle/working (no images): logits per model and split, the audit
arrays, metrics.json, and the exported EfficientNet-B0 (ONNX, ~16 MB).

Stages (each one saves its outputs before the next starts)
1. Audit: a 64-bit perceptual hash and a CLIP ViT-B/16 embedding for every image. For each test
   image: the nearest training image by CLIP cosine and the smallest pHash distance to any
   training image (near-duplicates leak answers). For each training image: the labels of its
   10 nearest training neighbours (label-noise candidates).
2. Models: CLIP zero-shot (descriptive prompts), a linear probe on frozen CLIP features, and
   fine-tuned ConvNeXt-Tiny (ImageNet-22k) and EfficientNet-B0 (timm). Best epoch on Validate.
3. Export: EfficientNet-B0 -> ONNX, checked against PyTorch with onnxruntime.
"""

from __future__ import annotations

import copy
import glob
import json
import math
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

SMOKE = int(os.environ.get("SMOKE", 0))  # >0: this many images per split (pipeline test)
OUT = Path("/kaggle/working")
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


SRC = Path(glob.glob("/kaggle/input/**/vnfood/classes.py", recursive=True)[0]).parent.parent
sys.path.insert(0, str(SRC))
from vnfood.classes import DISHES, prompt  # noqa: E402
from vnfood.metrics import expected_calibration_error, fit_temperature, softmax  # noqa: E402

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
IMAGES = Path(glob.glob("/kaggle/input/**/Images", recursive=True)[0])
CLASSES = sorted(p.name for p in (IMAGES / "Train").iterdir() if p.is_dir())
assert sorted(DISHES) == CLASSES, "class folders differ from vnfood.classes"
items: dict[str, list[tuple[str, int]]] = {}
for split in ("Train", "Validate", "Test"):
    rows = [
        (str(p.relative_to(IMAGES)), CLASSES.index(p.parent.name))
        for p in sorted((IMAGES / split).glob("*/*"))
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    if SMOKE:
        rows = rows[:: max(1, len(rows) // SMOKE)][:SMOKE]
    items[split] = rows
(OUT / "items.json").write_text(json.dumps({k: v for k, v in items.items()}))
log(device, {k: len(v) for k, v in items.items()}, f"{len(CLASSES)} classes")
metrics: dict = {"counts": {k: len(v) for k, v in items.items()}}


def save_metrics():
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load_rgb(rel: str, size: int = 320) -> Image.Image:
    img = Image.open(IMAGES / rel)
    img.draft("RGB", (size, size))  # JPEG DCT downscaling: much faster decoding of big photos
    return img.convert("RGB")


# --- 1. audit ------------------------------------------------------------------------------
def phash_of(rel: str) -> int:
    import imagehash

    try:
        return int(str(imagehash.phash(load_rgb(rel, 64))), 16)
    except Exception:
        return 0


all_rel = [r for s in ("Train", "Validate", "Test") for r, _ in items[s]]
with Pool(os.cpu_count()) as pool:
    hashes = np.array(pool.map(phash_of, all_rel, chunksize=64), dtype=np.uint64)
n_tr, n_va = len(items["Train"]), len(items["Validate"])
h_tr, h_te = hashes[:n_tr], hashes[n_tr + n_va :]
log("pHash done")

from transformers import CLIPModel, CLIPProcessor  # noqa: E402

clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device).eval().half()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")


class ImageList(torch.utils.data.Dataset):
    def __init__(self, rows, transform):
        self.rows, self.transform = rows, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        rel, y = self.rows[i]
        try:
            img = load_rgb(rel)
        except Exception:
            img = Image.new("RGB", (224, 224))
        return self.transform(img), y


def as_tensor(out) -> torch.Tensor:
    """get_*_features returns a tensor in transformers 4.x and an output object in some 5.x."""
    if torch.is_tensor(out):
        return out
    for key in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, key, None) is not None:
            return getattr(out, key)
    raise TypeError(f"unexpected CLIP output {type(out)}")


def clip_transform(img):
    return proc(images=img, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def clip_embed(model, rows) -> np.ndarray:
    dl = torch.utils.data.DataLoader(
        ImageList(rows, clip_transform), batch_size=128, num_workers=os.cpu_count(), pin_memory=True
    )
    out = []
    for x, _ in dl:
        f = as_tensor(model.get_image_features(pixel_values=x.to(device).half()))
        out.append(F.normalize(f.float(), dim=-1).cpu().numpy())
    return np.concatenate(out)


emb = {s: clip_embed(clip, items[s]) for s in ("Train", "Validate", "Test")}
y = {s: np.array([lab for _, lab in items[s]]) for s in items}
log("CLIP embeddings done")

# Nearest training neighbour of every test image, and pHash distance to the closest train image.
tr = torch.from_numpy(emb["Train"]).to(device)
te = torch.from_numpy(emb["Test"]).to(device)
cos, nn_idx = (te @ tr.T).max(dim=1)
min_ham = np.empty(len(h_te), dtype=np.int64)
pop = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
for s in range(0, len(h_te), 512):
    x = (h_te[s : s + 512, None] ^ h_tr[None, :]).view(np.uint8).reshape(-1, len(h_tr), 8)
    min_ham[s : s + 512] = pop[x].sum(axis=-1).min(axis=1)
# Label-noise candidates: labels of the 10 nearest training neighbours of each training image.
knn_labels = []
for s in range(0, len(tr), 1024):
    sims = tr[s : s + 1024] @ tr.T
    sims[torch.arange(sims.shape[0]), torch.arange(s, s + sims.shape[0])] = -1  # not itself
    knn_labels.append(torch.from_numpy(y["Train"]).to(device)[sims.topk(10, dim=1).indices].cpu())
np.savez_compressed(
    OUT / "audit.npz",
    test_nn_cos=cos.cpu().numpy(),
    test_nn_idx=nn_idx.cpu().numpy(),
    test_min_hamming=min_ham,
    train_knn_labels=torch.cat(knn_labels).numpy(),
    train_labels=y["Train"],
    test_labels=y["Test"],
)
log("audit saved")


def record(name: str, logits: dict[str, np.ndarray], extra: dict | None = None) -> None:
    """Save logits for Validate/Test and headline metrics (with temperature scaling)."""
    np.savez_compressed(OUT / f"logits_{name}.npz", **logits)
    t = fit_temperature(logits["Validate"], y["Validate"])
    out = {"temperature": t}
    for split in ("Validate", "Test"):
        p, p_t = softmax(logits[split]), softmax(logits[split], t)
        out[split] = {
            "top1": float((p.argmax(1) == y[split]).mean()),
            "top5": float((np.argsort(-p, 1)[:, :5] == y[split][:, None]).any(1).mean()),
            "ece": expected_calibration_error(p, y[split]),
            "ece_temperature": expected_calibration_error(p_t, y[split]),
        }
    metrics.setdefault("models", {})[name] = {**out, **(extra or {})}
    save_metrics()
    log(name, {k: round(v["top1"], 4) for k, v in out.items() if isinstance(v, dict)})


# --- 2a. CLIP zero-shot and linear probe ----------------------------------------------------
with torch.no_grad():
    tok = proc(text=[prompt(c) for c in CLASSES], return_tensors="pt", padding=True).to(device)
    text = F.normalize(as_tensor(clip.get_text_features(**tok)).float(), dim=-1).cpu().numpy()
scale = float(clip.logit_scale.exp())
record("CLIP zero-shot", {s: scale * emb[s] @ text.T for s in ("Validate", "Test")})

from sklearn.linear_model import LogisticRegression  # noqa: E402

best = (-1.0, None)
for c in (0.3, 1.0, 3.0, 10.0):
    lr = LogisticRegression(C=c, max_iter=2000).fit(emb["Train"], y["Train"])
    acc = lr.score(emb["Validate"], y["Validate"])
    best = max(best, (acc, c), key=lambda b: b[0])
probe = LogisticRegression(C=best[1], max_iter=2000).fit(emb["Train"], y["Train"])
record(
    "CLIP linear probe",
    {s: probe.decision_function(emb[s]) for s in ("Validate", "Test")},
    {"C": best[1]},
)
del clip
torch.cuda.empty_cache()

# --- 2b. fine-tuning ------------------------------------------------------------------------
import timm  # noqa: E402
from torchvision import transforms as T  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
train_tf = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ]
)
eval_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), T.Normalize(MEAN, STD)])


def loader(split, tf, shuffle):
    return torch.utils.data.DataLoader(
        ImageList(items[split], tf),
        batch_size=64,
        shuffle=shuffle,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=shuffle,
        persistent_workers=True,
    )


@torch.no_grad()
def predict(model, split) -> np.ndarray:
    model.eval()
    out = []
    for x, _ in loader(split, eval_tf, False):
        with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
            out.append(model(x.to(device, non_blocking=True)).float().cpu().numpy())
    return np.concatenate(out)


def finetune(arch: str, epochs: int, lr: float) -> torch.nn.Module:
    model = timm.create_model(arch, pretrained=True, num_classes=len(CLASSES)).to(device)
    model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    dl = loader("Train", train_tf, True)
    total = epochs * len(dl)
    warm = len(dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: (
            (s + 1) / warm
            if s < warm
            else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm)))
        ),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    best_acc, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t_ep, loss_sum, n = time.time(), 0.0, 0
        for x, yy in dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            yy = yy.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                loss = F.cross_entropy(model(x), yy, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            loss_sum += loss.detach().item() * len(yy)
            n += len(yy)
        val = predict(model, "Validate")
        acc = float((val.argmax(1) == y["Validate"]).mean())
        history.append(
            {"epoch": ep + 1, "loss": loss_sum / n, "val_top1": acc, "seconds": time.time() - t_ep}
        )
        log(
            f"{arch} epoch {ep + 1}: loss {loss_sum / n:.4f} val {acc:.4f} "
            f"({time.time() - t_ep:.0f}s)"
        )
        if acc > best_acc:
            best_acc, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    n_params = sum(p.numel() for p in model.parameters())
    record(
        arch,
        {s: predict(model, s) for s in ("Validate", "Test")},
        {
            "history": history,
            "parameters": n_params,
            "best_epoch": int(np.argmax([h["val_top1"] for h in history]) + 1),
        },
    )
    return model


EPOCHS = 2 if SMOKE else 10
finetune("convnext_tiny.fb_in22k_ft_in1k", EPOCHS, 2e-4)
torch.cuda.empty_cache()
b0 = finetune("efficientnet_b0.ra_in1k", EPOCHS, 1e-3)

# --- 3. ONNX export of the small model ------------------------------------------------------
b0 = b0.eval().float().to("cpu")
dummy = torch.randn(1, 3, 224, 224)
try:
    torch.onnx.export(
        b0,
        dummy,
        OUT / "efficientnet_b0.onnx",
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
except Exception as exc:  # keep every other output even if the export fails
    log("ONNX export failed", repr(exc))
(OUT / "classes.json").write_text(
    json.dumps(
        {
            "classes": CLASSES,
            "names": [DISHES[c][0] for c in CLASSES],
            "temperature": metrics["models"]["efficientnet_b0.ra_in1k"]["temperature"],
            "mean": MEAN,
            "std": STD,
        },
        ensure_ascii=False,
    )
)
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "onnxruntime"], check=True)
    import onnxruntime as ort

    sess = ort.InferenceSession(str(OUT / "efficientnet_b0.onnx"))
    x = torch.randn(4, 3, 224, 224)
    diff = np.abs(sess.run(None, {"image": x.numpy()})[0] - b0(x).detach().numpy()).max()
    t1 = time.time()
    for _ in range(20):
        sess.run(None, {"image": x[:1].numpy()})
    metrics["onnx"] = {
        "max_abs_diff_vs_pytorch": float(diff),
        "cpu_ms_per_image": 1000 * (time.time() - t1) / 20,
        "size_mb": os.path.getsize(OUT / "efficientnet_b0.onnx") / 1e6,
    }
    log("ONNX check", metrics["onnx"])
except Exception as exc:
    log("ONNX check failed", repr(exc))
save_metrics()
log("done")
