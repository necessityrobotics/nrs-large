#!/usr/bin/env python3
"""nrs-large on any stereo rig: rectify raw stereo images with the rig's calibration, then compute disparity and
metric depth (zero-shot: no training or fine-tuning for your camera).

  python rig_depth.py --ckpt nrs-large.safetensors --calib my_rig.json --left L.png --right R.png --out_dir out
  python rig_depth.py --ckpt ... --calib my_rig.json --sbs frame.png --out_dir out          # side-by-side image
  python rig_depth.py --ckpt ... --calib stereo.yml --left_dir L/ --right_dir R/ --out_dir out
  python rig_depth.py --ckpt ... --sequence recordings/rig1 --every 30 --out_dir out        # an xclean recording

Calibration (--calib):
  * an xclean calib.json (`xclean calib kalibr|opencv` writes it; an `xclean live --record` recording holds one):
    camera_width, camera_height, M1, D1, M2, D2, R_stereo, T_stereo (metres);
  * or an OpenCV FileStorage file (.yml / .yaml / .xml) from cv2.stereoCalibrate: K1, D1, K2, D2, R, T and
    image_width / image_height (or imageSize).  T is in the unit of the calibration target: --units mm or cm when it
    is not metres.
  R, T map points from the left camera to the right camera (cv2.stereoCalibrate's convention).  Pinhole lenses with
  OpenCV's distortion model (4, 5, 8, 12 or 14 coefficients).
Rectification: cv2.stereoRectify(alpha=0, CALIB_ZERO_DISPARITY) and cv2.initUndistortRectifyMap (float maps),
bilinear: the rectification of the example pairs of this repository.  Disparity and depth are given in the RECTIFIED
LEFT camera:
depth [m] = fx_rect [px] x baseline [m] / disparity [px].  The network is infer.py's (DLNR, 32 iterations, FP32).

Outputs in --out_dir, named after the left image (or the frame number of a sequence):
  <name>_disp.npy / _disp.png     float32 disparity (px) of the rectified left image / colourised
  <name>_depth.npy / _depth.png   float32 depth (m), NaN where disparity <= 0 / colourised inverse depth
  <name>_rect_left.png, <name>_rect_right.png   the rectified pair (with --save_rectified)
  <name>.ply                      a point cloud in the rectified left camera, metres (with --ply)
  rectification.json              fx, cx, cy, baseline, P1, P2, R1, R2, Q and the row-alignment check
A row-alignment check runs on the first pair: point tracks between the rectified images must lie on the same rows.
A median vertical offset above 1 px means the calibration does not fit the images (wrong file, swapped cameras,
a changed lens or focus).
"""
import argparse
import glob
import json
import os
import sys
import time

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

DIST_KEYS = ("D1", "D2")


class RigError(ValueError):
    """A calibration or input that cannot be used; the message says what to fix."""


# ================================================================================================= calibration
def _fs_get(fs, names):
    import numpy as np
    for n in names:
        node = fs.getNode(n)
        if node is not None and not node.empty():
            if node.isSeq():
                return np.array([node.at(i).real() for i in range(node.size())], float)
            if node.isReal() or node.isInt():
                return np.array([node.real()], float)
            m = node.mat()
            if m is not None:
                return np.asarray(m, float)
    return None


def load_calibration(path, units="m"):
    """-> dict(K1, D1, K2, D2, R, T (metres), size (W, H), source).  Raises RigError."""
    import cv2
    import numpy as np
    if not os.path.isfile(path):
        raise RigError("calibration file not found: %s" % path)
    if path.lower().endswith(".json"):
        try:
            c = json.load(open(path, encoding="utf-8"))
        except ValueError as e:
            raise RigError("%s is not valid JSON: %s" % (path, e))
        miss = [k for k in ("camera_width", "camera_height", "M1", "D1", "M2", "D2", "R_stereo", "T_stereo")
                if k not in c]
        if miss:
            raise RigError("%s is not an xclean calib.json: it has no %s" % (path, ", ".join(miss)))
        cal = {"K1": c["M1"], "D1": c["D1"], "K2": c["M2"], "D2": c["D2"], "R": c["R_stereo"],
               "T": np.asarray(c["T_stereo"], float), "size": (int(c["camera_width"]), int(c["camera_height"])),
               "source": "xclean calib.json"}
    else:
        fs = cv2.FileStorage(path, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise RigError("%s: OpenCV cannot read this file (expected an xclean calib.json or an OpenCV FileStorage "
                           ".yml / .yaml / .xml)" % path)
        try:
            cal = {}
            for key, names in (("K1", ("K1", "M1", "cameraMatrix1", "camera_matrix1")),
                               ("D1", ("D1", "distCoeffs1", "dist_coeffs1")),
                               ("K2", ("K2", "M2", "cameraMatrix2", "camera_matrix2")),
                               ("D2", ("D2", "distCoeffs2", "dist_coeffs2")),
                               ("R", ("R", "R_stereo")), ("T", ("T", "T_stereo"))):
                v = _fs_get(fs, names)
                if v is None:
                    raise RigError("%s has no %s (looked for %s)" % (path, key, ", ".join(names)))
                cal[key] = v
            w, h = _fs_get(fs, ("image_width", "imageWidth")), _fs_get(fs, ("image_height", "imageHeight"))
            if w is None or h is None:
                s = _fs_get(fs, ("imageSize", "image_size"))
                if s is None or s.size != 2:
                    raise RigError("%s has no image size (add image_width and image_height)" % path)
                w, h = s.ravel()[:1], s.ravel()[1:2]
            cal["size"] = (int(w.ravel()[0]), int(h.ravel()[0]))
        finally:
            fs.release()
        cal["T"] = np.asarray(cal["T"], float).reshape(3) * {"m": 1.0, "cm": 0.01, "mm": 0.001}[units]
        cal["source"] = "OpenCV FileStorage"
    for k, shape in (("K1", (3, 3)), ("K2", (3, 3)), ("R", (3, 3))):
        cal[k] = np.asarray(cal[k], float).reshape(shape)
    for k in DIST_KEYS:
        d = np.asarray(cal[k], float).ravel()
        if d.size not in (4, 5, 8, 12, 14):
            raise RigError("%s has %d coefficients; OpenCV's model takes 4, 5, 8, 12 or 14 (fisheye models are not "
                           "supported)" % (k, d.size))
        cal[k] = d
    cal["T"] = np.asarray(cal["T"], float).reshape(3)
    if not all(np.isfinite(cal[k]).all() for k in ("K1", "K2", "R", "T", "D1", "D2")):
        raise RigError("%s holds a value that is not finite" % path)
    b = float(np.linalg.norm(cal["T"]))
    if b > 5.0:
        unit = "millimetres" if b > 50 else "centimetres"
        if cal["source"] == "xclean calib.json":
            raise RigError("the baseline is %.4g 'metres': T_stereo looks like it is in %s, but a calib.json holds "
                           "metres (write it again with xclean calib)" % (b, unit))
        raise RigError("the baseline is %.4g 'metres': T looks like it is in %s (use --units %s)"
                       % (b, unit, "mm" if b > 50 else "cm"))
    if b < 0.005:
        raise RigError("the baseline is %.4g m: too small for stereo (wrong units, or the same camera twice?)" % b)
    return cal


def rectification(cal):
    """-> dict with the remap tables and the rectified geometry.  Raises RigError for a pair that is not a
    left / right pair."""
    import cv2
    import numpy as np
    W, H = cal["size"]
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(cal["K1"], cal["D1"], cal["K2"], cal["D2"], (W, H), cal["R"],
                                                      cal["T"].reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY,
                                                      alpha=0)
    if abs(P2[1, 3]) > abs(P2[0, 3]):
        raise RigError("this is a vertical pair (one camera above the other); the network needs a left / right pair")
    if P2[0, 3] > 0:
        raise RigError("the right camera lies to the LEFT of the left camera (T x is positive): swap the two images "
                       "(or the two cameras of the calibration)")
    m1 = cv2.initUndistortRectifyMap(cal["K1"], cal["D1"], R1, P1, (W, H), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(cal["K2"], cal["D2"], R2, P2, (W, H), cv2.CV_32FC1)
    fx = float(P1[0, 0])
    area = min(roi1[2] * roi1[3], roi2[2] * roi2[3]) / float(W * H)
    return {"maps": (m1, m2), "fx": fx, "cx": float(P1[0, 2]), "cy": float(P1[1, 2]),
            "baseline_m": float(-P2[0, 3] / P2[0, 0]), "size": [W, H], "P1": P1.tolist(), "P2": P2.tolist(),
            "R1": R1.tolist(), "R2": R2.tolist(), "Q": Q.tolist(), "valid_area": float(area)}


def rectify(rect, left, right):
    import cv2
    (m1x, m1y), (m2x, m2y) = rect["maps"]
    return (cv2.remap(left, m1x, m1y, cv2.INTER_LINEAR), cv2.remap(right, m2x, m2y, cv2.INTER_LINEAR))


def row_check(left, right):
    """Sub-pixel vertical offsets of left -> right point tracks on a rectified pair (Shi-Tomasi corners, pyramidal
    Lucas-Kanade with a forward-backward check; the check of the example pairs).  On a well rectified pair they are
    centred on 0.  -> {n, median_dy_px, median_abs_dy_px, p90_abs_dy_px}, or None with fewer than 20 tracks."""
    import cv2
    import numpy as np
    g1 = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY) if left.ndim == 3 else left
    g2 = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY) if right.ndim == 3 else right
    pts = cv2.goodFeaturesToTrack(g1, 3000, 0.01, 7)
    if pts is None:
        return None
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.01))
    fwd, st1, _ = cv2.calcOpticalFlowPyrLK(g1, g2, pts, None, **lk)
    bwd, st2, _ = cv2.calcOpticalFlowPyrLK(g2, g1, fwd, None, **lk)
    p0, p1, pb = pts.reshape(-1, 2), fwd.reshape(-1, 2), bwd.reshape(-1, 2)
    good = (st1.ravel() == 1) & (st2.ravel() == 1) & (np.linalg.norm(p0 - pb, axis=1) < 0.25)
    dx, dy = p0[:, 0] - p1[:, 0], p0[:, 1] - p1[:, 1]
    good &= np.abs(dy) < 8
    ok = good & (dx > -1) & (dx < 0.4 * g1.shape[1])                 # plausible stereo tracks only
    back = good & (dx < -1)                                          # the right image sees the scene to the LEFT
    swapped = bool(back.sum() >= 20 and back.sum() > 2 * ok.sum())
    if ok.sum() < 20:
        return {"n": int(ok.sum()), "swapped": swapped} if swapped else None
    d = dy[ok]
    return {"n": int(ok.sum()), "median_dy_px": float(np.median(d)), "median_abs_dy_px": float(np.median(np.abs(d))),
            "p90_abs_dy_px": float(np.percentile(np.abs(d), 90)), "swapped": swapped}


# ================================================================================================= inputs
def read_bgr(path):
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        raise RigError("cannot read %s" % path)
    return im


def pairs_from_args(a, size):
    """-> list of (name, left_path_or_None, right_path_or_None, sbs_path_or_None)."""
    out = []
    if a.left or a.right:
        if not (a.left and a.right):
            raise RigError("--left and --right go together")
        out.append((os.path.splitext(os.path.basename(a.left))[0], a.left, a.right, None))
    if a.sbs:
        for p in sorted(glob.glob(a.sbs)) or [a.sbs]:
            out.append((os.path.splitext(os.path.basename(p))[0], None, None, p))
    if a.left_dir or a.right_dir:
        if not (a.left_dir and a.right_dir):
            raise RigError("--left_dir and --right_dir go together")
        names = sorted(f for f in os.listdir(a.left_dir) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")))
        for f in names[::a.every]:
            r = os.path.join(a.right_dir, f)
            if not os.path.isfile(r):
                raise RigError("%s has no partner %s" % (os.path.join(a.left_dir, f), r))
            out.append((os.path.splitext(f)[0], os.path.join(a.left_dir, f), r, None))
    if a.sequence:
        ts = os.path.join(a.sequence, "timestamps.csv")
        if not os.path.isfile(ts):
            raise RigError("%s is not an xclean sequence folder (no timestamps.csv)" % a.sequence)
        ids = [int(ln.split(",")[0]) for ln in open(ts, encoding="utf-8").read().splitlines()[1:] if ln.strip()]
        for i in ids[::a.every]:
            for ext in ("png", "jpg"):
                lp = os.path.join(a.sequence, "left", "%06d.%s" % (i, ext))
                if os.path.isfile(lp):
                    out.append(("%06d" % i, lp, os.path.join(a.sequence, "right", "%06d.%s" % (i, ext)), None))
                    break
    if a.frames:
        out = out[:a.frames]
    if not out:
        raise RigError("no input: give --left/--right, --sbs, --left_dir/--right_dir or --sequence")
    return out


def load_pair(item, size, swap=False):
    import numpy as np
    name, lp, rp, sp = item
    if sp:
        im = read_bgr(sp)
        if im.shape[1] % 2:
            raise RigError("%s is %d px wide: a side-by-side image needs an even width" % (sp, im.shape[1]))
        w = im.shape[1] // 2
        left, right = np.ascontiguousarray(im[:, :w]), np.ascontiguousarray(im[:, w:])
    else:
        left, right = read_bgr(lp), read_bgr(rp)
    if swap:
        left, right = right, left
    W, H = size
    for side, im in (("left", left), ("right", right)):
        if (im.shape[1], im.shape[0]) != (W, H):
            raise RigError("the %s image of %s is %d x %d, but the calibration is %d x %d: give images at the "
                           "calibrated size" % (side, name, im.shape[1], im.shape[0], W, H))
    return left, right


# ================================================================================================= outputs
def write_ply(path, depth, rgb, rect, step=2):
    import numpy as np
    H, W = depth.shape
    v, u = np.mgrid[0:H:step, 0:W:step]
    z = depth[::step, ::step]
    ok = np.isfinite(z) & (z > 0)
    x = (u - rect["cx"]) * z / rect["fx"]
    y = (v - rect["cy"]) * z / rect["fx"]
    c = rgb[::step, ::step][ok].astype(np.uint8)
    pts = np.stack([x[ok], y[ok], z[ok]], 1).astype(np.float32)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\ncomment nrs-large rig_depth.py: rectified left camera, "
                 "metres, x right, y down, z forward\nelement vertex %d\nproperty float x\nproperty float y\n"
                 "property float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
                 % len(pts)).encode("ascii"))
        rec = np.empty(len(pts), dtype=[("p", "<f4", 3), ("c", "u1", 3)])
        rec["p"], rec["c"] = pts, c
        f.write(rec.tobytes())
    return len(pts)


# ================================================================================================= main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 epilog="Guide: ANY_STEREO_RIG.md. Model card: MODEL_CARD.md.")
    ap.add_argument("--ckpt", required=True, help="nrs-large.safetensors")
    ap.add_argument("--calib", required=False, default=None,
                    help="the rig's calibration: an xclean calib.json or an OpenCV .yml/.xml (with --sequence, the "
                         "default is the sequence's calib.json)")
    ap.add_argument("--units", choices=("m", "cm", "mm"), default="m",
                    help="the unit of T in an OpenCV file (default m; an xclean calib.json is always metres)")
    ap.add_argument("--left", default=None, help="the raw left image")
    ap.add_argument("--right", default=None, help="the raw right image")
    ap.add_argument("--sbs", default=None,
                    help="a side-by-side image (left camera in the left half), or a glob pattern of them")
    ap.add_argument("--left_dir", default=None, help="a folder of raw left images")
    ap.add_argument("--right_dir", default=None, help="a folder of raw right images with the same file names")
    ap.add_argument("--sequence", default=None, help="an xclean sequence folder (for example an xclean live recording)")
    ap.add_argument("--every", type=int, default=1, help="with folders: use every N-th pair (default 1)")
    ap.add_argument("--frames", type=int, default=None, help="use at most N pairs")
    ap.add_argument("--swap", action="store_true", help="the left image is the right camera: swap the two")
    ap.add_argument("--rectified", action="store_true",
                    help="the images are already rectified: skip rectification (no --calib; give --fx and "
                         "--baseline for depth; the .ply then takes the image centre as the principal point)")
    ap.add_argument("--fx", type=float, default=None, help="with --rectified: the rectified focal length, pixels")
    ap.add_argument("--baseline", type=float, default=None, help="with --rectified: the baseline, metres")
    ap.add_argument("--out_dir", default="out")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="default cuda")
    ap.add_argument("--iters", type=int, default=32, help="update iterations (default 32, as the model card)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads")
    ap.add_argument("--save_rectified", action="store_true", help="also write the rectified pair")
    ap.add_argument("--ply", action="store_true",
                    help="also write a point cloud (.ply) of every pair (needs depth: a calibration, or --rectified "
                         "with --fx and --baseline)")
    ap.add_argument("--max_depth", type=float, default=None,
                    help="set depth beyond this many metres to NaN (default: keep every value)")
    a = ap.parse_args(argv)
    if a.every < 1:
        ap.error("--every must be 1 or more")
    if a.frames is not None and a.frames < 1:
        ap.error("--frames must be 1 or more")
    if a.rectified and a.calib:
        ap.error("--rectified takes no --calib: give --fx and --baseline of the rectified pair")
    if not a.rectified and (a.fx is not None or a.baseline is not None):
        ap.error("--fx and --baseline go with --rectified; with a calibration they come from the file")
    import cv2
    import numpy as np
    try:
        if a.rectified:
            if (a.fx is None) != (a.baseline is None):
                ap.error("--fx and --baseline go together")
            first = pairs_from_args(a, None)
            im = read_bgr(first[0][3] or first[0][1])
            W = im.shape[1] // 2 if first[0][3] else im.shape[1]
            rect = {"fx": a.fx, "baseline_m": a.baseline, "cx": W / 2.0, "cy": im.shape[0] / 2.0,
                    "size": [W, im.shape[0]], "maps": None, "rectified_input": True}
            size = tuple(rect["size"])
        else:
            cpath = a.calib or (os.path.join(a.sequence, "calib.json") if a.sequence else None)
            if cpath is None:
                ap.error("give --calib (the rig's calibration), or --rectified for an already rectified pair")
            cal = load_calibration(cpath, a.units)
            rect = rectification(cal)
            rect["calibration"] = {"file": os.path.abspath(cpath), "format": cal["source"],
                                   "distortion_coefficients": int(cal["D1"].size)}
            size = cal["size"]
            first = pairs_from_args(a, size)
    except RigError as e:
        print("rig_depth: error: %s" % e, file=sys.stderr)
        return 2
    os.makedirs(a.out_dir, exist_ok=True)
    if a.rectified and a.fx is None:
        print("rig_depth: note: no --fx and --baseline: disparity only (no depth, no .ply)")
    print("rig: %d x %d, rectified focal %.2f px, baseline %.4f m%s" % (
        size[0], size[1], rect["fx"] or 0, rect["baseline_m"] or 0,
        "" if a.rectified else ", valid area after rectification %.0f %%" % (100 * rect["valid_area"])))
    if not a.rectified and rect["valid_area"] < 0.5:
        print("rig_depth: warning: only %.0f %% of the image is valid after rectification; check the calibration"
              % (100 * rect["valid_area"]))

    import torch
    from infer import build, colourise, infer
    torch.set_num_threads(a.threads)
    if a.device == "cuda":
        if not torch.cuda.is_available():
            print("rig_depth: error: no CUDA GPU; use --device cpu", file=sys.stderr)
            return 2
        torch.backends.cuda.matmul.allow_tf32 = False                # FP32 as the model card's numbers
        torch.backends.cudnn.allow_tf32 = False
    t0 = time.perf_counter()
    model, finfo = build(a.ckpt, a.device, a.iters)
    print("model: nrs-large (DLNR network design by Zhao et al., MIT code; weights by Necessity Robotics), %.2f M "
          "params, weights %s..., loaded in %.1f s" % (finfo["params"] / 1e6, finfo["weights_fingerprint"][:12],
                                                     time.perf_counter() - t0))
    meta = {k: v for k, v in rect.items() if k != "maps"}
    meta.update(weights=finfo, iters=a.iters, device=a.device,
                depth_frame="the rectified left camera: x right, y down, z forward, metres",
                depth_formula="depth = fx * baseline / disparity")
    n_done = 0
    try:
        for k, item in enumerate(first):
            left, right = load_pair(item, size, a.swap)
            if not a.rectified:
                left, right = rectify(rect, left, right)
            if k == 0:
                meta["row_check"] = row_check(left, right)
                rc = meta["row_check"]
                if rc is not None and rc.get("swapped"):
                    print("rig_depth: warning: the left and right images look swapped: the right image sees the "
                          "scene shifted the wrong way. Swap them (--swap, or swap the two files) and run again.")
                if rc is None or "median_abs_dy_px" not in rc:
                    print("row check: too few point tracks to check the rectification")
                else:
                    print("row check (first pair): %d point tracks, median vertical offset %.2f px, 90 %% below %.2f px"
                          % (rc["n"], rc["median_abs_dy_px"], rc["p90_abs_dy_px"]))
                    if rc["median_abs_dy_px"] > 1.0:
                        print("rig_depth: warning: the rows of the rectified pair do not line up (median offset "
                              "%.2f px > 1 px). The calibration does not fit these images: check the file, the "
                              "left / right order (--swap) and the image size." % rc["median_abs_dy_px"])
            name = item[0]
            if a.save_rectified:
                cv2.imwrite(os.path.join(a.out_dir, name + "_rect_left.png"), left)
                cv2.imwrite(os.path.join(a.out_dir, name + "_rect_right.png"), right)
            L = cv2.cvtColor(left, cv2.COLOR_BGR2RGB).astype(np.float32)
            R = cv2.cvtColor(right, cv2.COLOR_BGR2RGB).astype(np.float32)
            disp, info = infer(model, L, R, a.iters, a.device)
            np.save(os.path.join(a.out_dir, name + "_disp.npy"), disp)
            cv2.imwrite(os.path.join(a.out_dir, name + "_disp.png"), colourise(disp, 0.0, None))
            line = "%s: %.2f s, disparity median %.2f px" % (name, info["s"], float(np.median(disp)))
            if rect["fx"] and rect["baseline_m"]:
                depth = np.full(disp.shape, np.nan, np.float32)
                ok = disp > 0
                depth[ok] = np.float32(rect["fx"] * rect["baseline_m"]) / disp[ok]
                if a.max_depth:
                    depth[depth > a.max_depth] = np.nan
                np.save(os.path.join(a.out_dir, name + "_depth.npy"), depth)
                fin = np.isfinite(depth)
                cv2.imwrite(os.path.join(a.out_dir, name + "_depth.png"),
                            colourise(np.where(fin, 1.0 / np.maximum(depth, 1e-6), np.nan)))
                if fin.any():
                    line += ", depth median %.2f m (5-95 %%: %.2f-%.2f m)" % (
                        float(np.median(depth[fin])), float(np.percentile(depth[fin], 5)),
                        float(np.percentile(depth[fin], 95)))
                if a.ply:
                    write_ply(os.path.join(a.out_dir, name + ".ply"), depth, L, rect)
            if not info["refinement_applied"]:
                line += " (refinement skipped: the authors' rule)"
            print(line, flush=True)
            n_done += 1
    except RigError as e:
        print("rig_depth: error: %s" % e, file=sys.stderr)
        return 2
    finally:
        meta["pairs_done"] = n_done
        json.dump(meta, open(os.path.join(a.out_dir, "rectification.json"), "w", encoding="utf-8"), indent=1)
    print("wrote %d pair(s) to %s" % (n_done, a.out_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
