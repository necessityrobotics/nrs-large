# nrs-large on any stereo rig

**nrs-large is zero-shot.** It never saw your camera. It never saw a real camera image in training. We trained it
only on computer-made (synthetic) stereo pairs: WMGStereo-150k, 155,769 rendered pairs. Then we tested it zero-shot
on six real benchmark sets. It passed all 4 accuracy bars of its pre-registered release gate.

So you do not train or fine-tune nrs-large for your rig. Calibrate the rig, rectify the images, run the model.
`rig_depth.py` does the rectification and the inference in one command.

The network design is not ours. DLNR (High-Frequency Stereo Matching Network) is by Zhao et al., CVPR 2023, and its
code is MIT-licensed. We trained the weights from random initialisation. The weights and our scripts are
Apache-2.0. Credits: `NOTICE`, `ATTRIBUTION.md`.

## The zero-shot results

Each benchmark was scored once, with the weights of the release. None of these sets was in the training data. Lower
is better (the percentage of pixels with a large disparity error).

| Benchmark (real images) | nrs-large 0.1.0 | Pre-registered bar |
|---|---|---|
| Middlebury 2014, half resolution, bad-2 (non-occluded) | **3.91 %** | 5.82 % or lower: passed |
| Middlebury 2021, bad-2 | **8.44 %** | no bar |
| ETH3D, bad-1 | **2.04 %** | 3.37 % or lower: passed |
| KITTI 2012, bad-3 | **3.38 %** | no bar |
| KITTI 2015, bad-3 | **4.59 %** | 5.53 % or lower: passed |
| Booster, quarter resolution, bad-2 | **9.22 %** | 11.9 % or lower: passed |

Details, the published reference row and the known gaps (for example Middlebury 2021: 8.44 % for us, 6.72 % in the
published DLNR row) are in `MODEL_CARD.md`.

**What zero-shot does not mean.** It does not mean every scene works. The model learnt from synthetic images only:
real camera noise, low light and lens flare were never in training. Glass, mirrors, screens, blank walls and thin
structures are hard. Every pixel gets a value, also where the value is a guess; there is no confidence map.

## What you need

- A stereo rig: two cameras side by side, looking the same way, with one image size.
- Pinhole lenses with OpenCV's distortion model (4, 5, 8, 12 or 14 coefficients). Fisheye models are not supported.
- The two images of a pair taken at the same moment (a hardware trigger) if the rig or the scene moves.
- The install of `README.md` (PyTorch, `requirements.txt`, the weights).

## Step 1: calibrate the rig

`rig_depth.py` reads two calibration formats:

| format | made by | `--calib` |
|---|---|---|
| xclean `calib.json` | `xclean calib opencv` or `xclean calib kalibr` (xclean, Apache-2.0); an `xclean live --record` recording holds one | `--calib my_rig.json` |
| OpenCV FileStorage (`.yml`, `.yaml`, `.xml`) | your own `cv2.stereoCalibrate` script: `K1`, `D1`, `K2`, `D2`, `R`, `T`, `image_width`, `image_height` | `--calib stereo.yml` (add `--units mm` or `--units cm` when `T` is not in metres) |

`R` and `T` map a point from the left camera into the right camera, as `cv2.stereoCalibrate` gives them.

If you have no calibration yet, use the chessboard procedure of xclean 0.3.0 or later: capture image pairs with
`xclean calib capture`, calibrate them with the short OpenCV script, and let `xclean calib opencv` check the result.
The steps are in xclean's guide, `docs/ANY_STEREO_LIVE.md`, "Step 1: calibrate"
(<https://github.com/necessityrobotics/xclean>).

> **CAUTION:** Calibrate at the image size that you give to the model. A calibration of 1280 x 720 does not fit
> images of 640 x 360. `rig_depth.py` stops with a message when the sizes differ.

## Step 2: take the images

Use any tool that saves the two images of a pair: the camera's own software, OpenCV, or `xclean live --record`. Give
the **raw** images, as the cameras took them. `rig_depth.py` rectifies them with the calibration.

## Step 3: run

One pair:

```
python rig_depth.py --ckpt nrs-large.safetensors --calib my_rig.json --left left.png --right right.png --out_dir out
```

A side-by-side image (the left camera in the left half):

```
python rig_depth.py --ckpt nrs-large.safetensors --calib my_rig.json --sbs frame.png --out_dir out
```

Two folders of images with the same file names:

```
python rig_depth.py --ckpt nrs-large.safetensors --calib stereo.yml --left_dir left --right_dir right --out_dir out
```

An xclean recording (the recording's own `calib.json` is used; `--every 30` takes one pair per second at 30 Hz):

```
python rig_depth.py --ckpt nrs-large.safetensors --sequence recordings/rig1 --every 30 --out_dir out
```

- `--device cpu` runs without a CUDA GPU (slow: about 6.5 s per 640 x 480 pair on 8 CPU threads; 0.28 s on an
  RTX 5090). These are network times after warm-up: the first pair on CUDA takes longer (0.9 s to 5.5 s in our
  runs). nrs-large is not for real-time use.
- `--swap` swaps the two images when the left image is from the right camera.
- `--save_rectified` also writes the rectified pair. `--ply` also writes a point cloud.
- `--rectified --fx F --baseline B` takes pairs that are already rectified (it skips step 1). The `.ply` then
  takes the image centre as the principal point.

## Step 4: check the result

At start, `rig_depth.py` prints the rectified geometry and a row check of the first pair. Our OAK-D Lite gave:

```
rig: 640 x 400, rectified focal 471.05 px, baseline 0.0751 m, valid area after rectification 100 %
row check (first pair): 2343 point tracks, median vertical offset 0.24 px, 90 % below 0.44 px
```

- After rectification, a point must lie on the same row in both images. A median vertical offset below about 0.5 px
  is good.
- Above 1 px, the script prints a warning: the calibration does not fit the images. Check the calibration file, the
  left / right order (`--swap`) and the image size. A changed focus or zoom also breaks a calibration.
- When the left and right images are swapped, the script prints a warning that says so. Swap them and run again.
- The baseline must be the distance between the two lenses. Measure it with a ruler to check.

## Outputs

All outputs are in the **rectified left camera**: x to the right, y down, z forward, metres.

| file | content |
|---|---|
| `<name>_disp.npy` | float32 disparity, pixels |
| `<name>_disp.png` | colourised disparity |
| `<name>_depth.npy` | float32 depth in metres, NaN where the disparity is 0 or less: depth = focal x baseline / disparity |
| `<name>_depth.png` | colourised inverse depth |
| `<name>_rect_left.png`, `<name>_rect_right.png` | the rectified pair (with `--save_rectified`) |
| `<name>.ply` | a coloured point cloud, every second pixel (with `--ply`) |
| `rectification.json` | focal length, principal point, baseline, `P1`, `P2`, `R1`, `R2`, `Q`, the row check |

`R1` in `rectification.json` rotates a point of the raw left camera into the rectified left camera. The transpose
of `R1` rotates a point of the outputs back into the raw left camera axes.

## Known limits

- Synthetic-only training (see "What zero-shot does not mean").
- Depth error grows with distance: one pixel of disparity error changes far depth much more than near depth.
- No embedded (Jetson) timing. The model is for desktops and servers.
- Not for safety-critical use without an independent safety layer.
- We tested `rig_depth.py` on our own OAK-D Lite recordings and on calibration files of the formats above. We did not
  test other stereo cameras.
