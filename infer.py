#!/usr/bin/env python3
"""nrs-large inference (DLNR, PyTorch): disparity, and optionally metric depth, from one rectified stereo pair.

  python infer.py --ckpt nrs-large.safetensors --left L.png --right R.png --out_dir out/ [--device cuda]
  python infer.py ... --fx 471.049 --baseline 0.07514                  # also writes depth in metres

Model: the authors' DLNR code (MIT; github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network, commit
c10475f), used read-only from the unmodified copy in third_party/DLNR next to this file, or from the folder that
the NRS_DLNR_DIR environment variable names. Settings = the training configuration (DLNR model arguments): 'reg'
correlation (pure PyTorch, no CUDA extension), 4 levels, radius 4, 3 GRU levels of 128, 1/4 resolution; FP32;
32 update iterations. Input padded to a multiple of 32 with the authors' InputPadder. The network outputs flow;
disparity = -flow.

Outputs in --out_dir, named after the left image:
  <name>_disp.npy  float32 HxW disparity (px)      <name>_disp.png  colourised disparity
  <name>_depth.npy float32 HxW depth (m), NaN where disparity <= 0   <name>_depth.png  (with --fx and --baseline)

Speed (network forward pass, 640x480, 32 iterations, FP32, after warm-up; measured 2026-11-12): about 6.5 s per
pair on 8 CPU threads (Ryzen 9 9950X) and 0.28 s on an RTX 5090; use --device cuda for real work.
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.dont_write_bytecode = True                       # nothing is written into the read-only authors' tree
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def _find_dlnr():
    d = os.environ.get("NRS_DLNR_DIR") or os.path.join(HERE, "third_party", "DLNR")
    if not os.path.isfile(os.path.join(d, "core", "dlnr.py")):
        sys.exit("DLNR code not found in %s: use the third_party/DLNR folder of this repository or set NRS_DLNR_DIR" % d)
    return d


DLNR_DIR = _find_dlnr()
DLNR_COMMIT = "c10475fe30ff198781ecc7c712c24c781ff05096"


def dlnr_args(iters=32):
    """The training model arguments with mixed_precision off (FP32 inference); train-only fields are unused here."""
    return argparse.Namespace(
        mixed_precision=False, corr_implementation="reg", shared_backbone=False, corr_levels=4, corr_radius=4,
        n_downsample=2, slow_fast_gru=False, n_gru_layers=3, hidden_dims=[128] * 3, train_iters=22,
        valid_iters=iters, image_size=[384, 736], batch_size=8, spatial_scale=[-0.2, 0.4],
        saturation_range=[0.0, 1.4], img_gamma=None, do_flip=False, noyjitter=False,
        train_datasets=["wmgstereo"])  # unused at inference


def build(ckpt, device="cpu", iters=32):
    import torch
    from weights_util import load_state, weights_fingerprint
    if DLNR_DIR not in sys.path:
        sys.path.insert(0, DLNR_DIR)
    from core.dlnr import DLNR
    sd, finfo = load_state(ckpt)
    model = DLNR(dlnr_args(iters))
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    finfo["weights_fingerprint"] = weights_fingerprint(sd)
    finfo["params"] = int(sum(p.numel() for p in model.parameters()))
    return model, finfo


def read_rgb(path):
    import cv2
    import numpy as np
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)


def colourise(values, vmin=None, vmax=None):
    import cv2
    import numpy as np
    v = values.astype(np.float32)
    ok = np.isfinite(v)
    if not ok.any():
        return np.zeros(v.shape + (3,), np.uint8)
    lo = np.percentile(v[ok], 1) if vmin is None else vmin
    hi = np.percentile(v[ok], 99) if vmax is None else vmax
    x = np.clip((np.where(ok, v, lo) - lo) / max(hi - lo, 1e-6), 0, 1)
    out = cv2.applyColorMap((x * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    out[~ok] = 0
    return out


def infer(model, left, right, iters=32, device="cpu"):
    """left/right: HxWx3 float32 RGB 0-255. Returns (disparity HxW float32, info dict)."""
    import numpy as np
    import torch
    from core.utils.utils import InputPadder
    warnings.filterwarnings("ignore", category=FutureWarning)       # torch.cuda.amp.autocast deprecation in DLNR
    warnings.filterwarnings("ignore", message=".*grid_sample and affine_grid behavior.*")   # DLNR refinement warp
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")                       # DLNR coordinate grids
    i1 = torch.from_numpy(left).permute(2, 0, 1)[None].to(device)
    i2 = torch.from_numpy(right).permute(2, 0, 1)[None].to(device)
    pad = InputPadder(i1.shape, divis_by=32)
    i1, i2 = pad.pad(i1, i2)
    hit = []
    h = model.normalizationRefinement.register_forward_hook(lambda *_: hit.append(1))
    try:
        t0 = time.perf_counter()
        with torch.no_grad():
            _, flow = model(i1, i2, iters=iters, test_mode=True)
        if device != "cpu":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
    finally:
        h.remove()
    disp = (-pad.unpad(flow.float())[0, 0]).cpu().numpy().astype(np.float32)
    return disp, {"s": round(dt, 2), "iters": iters, "padded_hw": list(i1.shape[-2:]),
                  "refinement_applied": bool(hit)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ckpt", required=True, help=".pth (DataParallel keys are fine) or .safetensors")
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--fx", type=float, default=None, help="rectified focal length, pixels")
    ap.add_argument("--baseline", type=float, default=None, help="stereo baseline, metres")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--iters", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8, help="CPU threads")
    ap.add_argument("--vmax", type=float, default=None, help="disparity at the top of the colour map (default p99)")
    ap.add_argument("--json", default=None, help="write run info here")
    a = ap.parse_args()
    if (a.fx is None) != (a.baseline is None):
        ap.error("--fx and --baseline go together")

    import cv2
    import numpy as np
    import torch
    torch.set_num_threads(a.threads)
    if a.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False                # keep FP32 FP32
        torch.backends.cudnn.allow_tf32 = False
    L, R = read_rgb(a.left), read_rgb(a.right)
    if L.shape != R.shape:
        sys.exit("left %s and right %s differ in size" % (L.shape, R.shape))
    t0 = time.perf_counter()
    model, finfo = build(a.ckpt, a.device, a.iters)
    print("model: DLNR, %.2f M params, weights %s... (%s), loaded in %.1f s" % (
        finfo["params"] / 1e6, finfo["weights_fingerprint"][:12], os.path.basename(a.ckpt), time.perf_counter() - t0))
    disp, info = infer(model, L, R, a.iters, a.device)
    print("inference: %.1f s on %s, %d iterations, padded to %s, refinement %s"
          % (info["s"], a.device, a.iters, "x".join(map(str, info["padded_hw"][::-1])),
             "applied" if info["refinement_applied"] else "SKIPPED (some flow >= 0, the authors' rule)"))
    print("disparity: min %.2f  median %.2f  max %.2f px" % (disp.min(), np.median(disp), disp.max()))

    os.makedirs(a.out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(a.left))[0]
    for name in ("_disp.npy", "_disp.png", "_depth.npy", "_depth.png"):
        p = os.path.join(a.out_dir, stem + name)
        if os.path.exists(p):
            os.remove(p)
    np.save(os.path.join(a.out_dir, stem + "_disp.npy"), disp)
    cv2.imwrite(os.path.join(a.out_dir, stem + "_disp.png"), colourise(disp, 0.0, a.vmax))
    if a.fx is not None:
        depth = np.full(disp.shape, np.nan, np.float32)
        ok = disp > 0
        depth[ok] = np.float32(a.fx * a.baseline) / disp[ok]
        np.save(os.path.join(a.out_dir, stem + "_depth.npy"), depth)
        cv2.imwrite(os.path.join(a.out_dir, stem + "_depth.png"), colourise(np.where(ok, 1.0 / np.maximum(depth, 1e-6), np.nan)))
        if ok.any():
            print("depth: median %.2f m, 5-95 %% range %.2f-%.2f m, %.1f %% valid"
                  % (np.median(depth[ok]), np.percentile(depth[ok], 5), np.percentile(depth[ok], 95), 100 * ok.mean()))
    if a.json:
        json.dump({"weights": finfo, "run": info, "left": os.path.abspath(a.left), "right": os.path.abspath(a.right),
                   "device": a.device, "threads": a.threads, "torch": torch.__version__, "dlnr_dir": DLNR_DIR,
                   "dlnr_commit_expected": DLNR_COMMIT,
                   "disparity": {"min": float(disp.min()), "median": float(np.median(disp)), "max": float(disp.max())}},
                  open(a.json, "w", encoding="utf-8"), indent=1)
    print("wrote %s_disp.npy/.png%s in %s" % (stem, " and _depth.npy/.png" if a.fx is not None else "", a.out_dir))


if __name__ == "__main__":
    main()
