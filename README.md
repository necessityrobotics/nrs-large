# nrs-large

`nrs-large` is a stereo depth network by Necessity Robotics. It takes a rectified stereo pair and gives the disparity of the left image, and metric depth if you give the camera's focal length and baseline.

**Zero-shot, on any stereo rig.** We trained nrs-large only on computer-made (synthetic) stereo pairs and tested it zero-shot on six real benchmark sets. You do not train it for your camera: calibrate the rig, and `rig_depth.py` rectifies the raw images and runs the model. Guide: [`ANY_STEREO_RIG.md`](ANY_STEREO_RIG.md).

- **Network:** DLNR (High-Frequency Stereo Matching Network, CVPR 2023), the authors' MIT code, unmodified.
- **Weights:** ours, trained **from random initialisation** on WMGStereo-150k only (200,000 steps). No other group's weights were used.
- **Licence:** Apache-2.0 for our weights and scripts. The DLNR code in `third_party/DLNR` keeps its MIT licence. Credits: `NOTICE`, `ATTRIBUTION.md`.
- **Weights download:** Hugging Face, [`Pyassahu/nrs-large`](https://huggingface.co/Pyassahu/nrs-large).
- **Model card:** `MODEL_CARD.md` (accuracy, speed, training data, limits).

## Accuracy (zero-shot, each benchmark scored once)

Lower is better. WMG protocol: Middlebury 2014 non-occluded, all other sets all pixels, mean over images. IEEE FP32.

| Model | Middlebury-H bad-2 | Middlebury 2021 bad-2 | ETH3D bad-1 | KITTI-12 bad-3 | KITTI-15 bad-3 | Booster-Q bad-2 |
|---|---|---|---|---|---|---|
| **nrs-large 0.1.0** | **3.91** | **8.44** | **2.04** | **3.38** | **4.59** | **9.22** |
| DLNR on WMGStereo-150k, published (WMG paper, CVPR 2026, Table 2) | 3.76 | 6.72 | 2.50 | 3.30 | 4.54 | 9.09 |

The published row is the WMG authors' number, not scored by us. See `MODEL_CARD.md` for the full table, the reference row and the known gaps.

## Install

You need Python 3.10 or newer. A CUDA GPU is strongly recommended (CPU works, but slowly).

1. Install PyTorch for your system from https://pytorch.org (CUDA or CPU build).
2. Install the other packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the weights into this folder:

   ```bash
   pip install huggingface_hub
   hf download Pyassahu/nrs-large nrs-large.safetensors nrs-large.safetensors.sha256 --local-dir .
   ```

4. Check the download. On Linux, the command must print `nrs-large.safetensors: OK` (on macOS use `shasum -a 256 -c nrs-large.safetensors.sha256`):

   ```bash
   sha256sum -c nrs-large.safetensors.sha256
   ```

   On Windows (PowerShell), compare the printed hash with the one in `nrs-large.safetensors.sha256`:

   ```powershell
   Get-FileHash nrs-large.safetensors -Algorithm SHA256
   ```

## Run

```bash
python infer.py --ckpt nrs-large.safetensors --left examples/pair_01_left.png --right examples/pair_01_right.png --out_dir out --device cuda --fx 471.049 --baseline 0.07514
```

- `--fx` is the rectified focal length in pixels and `--baseline` the stereo baseline in metres. These two values are for the example pairs (an OAK-D Lite). Without them you get disparity only.
- Use `--device cpu` if you have no CUDA GPU.

**Outputs** in `--out_dir`, named after the left image:

| File | Content |
|---|---|
| `<name>_disp.npy` | float32 disparity of the left image, pixels |
| `<name>_disp.png` | colourised disparity |
| `<name>_depth.npy` | float32 depth in metres, NaN where disparity ≤ 0 (with `--fx` and `--baseline`) |
| `<name>_depth.png` | colourised inverse depth (with `--fx` and `--baseline`) |

**Your own images:** `infer.py` needs a **rectified** stereo pair (rows aligned), left and right of the same size. Any size works; the script pads to a multiple of 32 and crops back.

## Any stereo rig

Raw images from your own stereo camera: give the rig's calibration (an xclean `calib.json` or an OpenCV `cv2.stereoCalibrate` file), and `rig_depth.py` rectifies the pair, runs the model and writes disparity and metric depth:

```bash
python rig_depth.py --ckpt nrs-large.safetensors --calib my_rig.json --left left.png --right right.png --out_dir out
```

It also takes side-by-side images (`--sbs`), folders of pairs (`--left_dir`, `--right_dir`) and xclean recordings (`--sequence`). It checks the rectification of the first pair and warns when the calibration does not fit the images. Step-by-step guide (calibrate, capture, run, check): [`ANY_STEREO_RIG.md`](ANY_STEREO_RIG.md).

**Speed** (network forward pass, 640x480, 32 iterations, FP32, after warm-up): 0.28 s per pair on an RTX 5090; about 6.5 s per pair on 8 CPU threads (Ryzen 9 9950X). The first pair in a new process is slower on CUDA.

## Known limits

- Trained on synthetic data only. Real camera noise and low light were never seen in training.
- Every pixel gets a depth, also where the answer is a guess (sky, glass, mirrors, blank walls). There is no confidence map.
- Hard cases: glass, mirrors, screens, textureless or dark surfaces, thin structures.
- Not for safety-critical use without an independent safety layer.

See `MODEL_CARD.md` for the measured details.

## Folder contents

| Path | What |
|---|---|
| `infer.py`, `weights_util.py` | inference script and weight loading (Apache-2.0) |
| `rig_depth.py`, `ANY_STEREO_RIG.md` | any calibrated stereo rig: rectification, then inference; the guide (Apache-2.0) |
| `third_party/DLNR/` | the DLNR authors' code, commit c10475f, unmodified subset (MIT) |
| `examples/` | three rectified pairs from our own OAK-D Lite (640x400) |
| `MODEL_CARD.md`, `ATTRIBUTION.md`, `NOTICE`, `LICENSE`, `LICENSES/` | card, credits and licence texts |

## Citation and credits

The network design and code are by Zhao et al., *High-Frequency Stereo Matching Network*, CVPR 2023. Their code builds on RAFT-Stereo (Lipson, Teed, Deng, 3DV 2021). The training data is WMGStereo-150k (Yan, Raistrick, Deng, *What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?*, CVPR 2026, arXiv 2504.16930). Please cite all three.

Contact: necessityrobotics@gmail.com
