# Model card: nrs-large 0.1.0

**Summary.** `nrs-large` is the larger, slower stereo network of the Necessity Robotics robot stack. It runs on desktops and servers. It is meant to label real stereo footage for `nrs-edge` (the planned on-board model) and to give depth for maps built off the robot. It takes a rectified stereo pair and returns the disparity of the left image. It is zero-shot: it was trained only on synthetic stereo pairs, and it was tested on six real benchmark sets that were not in its training data. It needs no training for a new camera; `rig_depth.py` rectifies the raw images of any calibrated stereo rig and runs it.

| | |
|---|---|
| Maker | Necessity Robotics (a robotics company; this network is the teacher and map-depth part of its robot stack) |
| Architecture | DLNR (High-Frequency Stereo Matching Network): channel-attention transformer feature extractor; 1-D correlation pyramid (4 levels, radius 4); multi-level LSTM-style update block (3 levels of 128); normalization refinement. Iterative: 32 updates at inference. |
| Code | The authors' MIT code, github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network, commit c10475f. `reg` correlation (pure PyTorch). |
| Weights | **Ours, trained from random initialisation** (the extractor loads no pretrained weights). The authors' DLNR weights were not used. |
| Parameters | 57.38 M (57,378,681) |
| Licence | **Apache-2.0** (weights and scripts; see `LICENSE`). Third-party code and data credits: `NOTICE`, `ATTRIBUTION.md`. |
| Release files | `nrs-large.safetensors` (236,755,204 bytes, sha256 `9ae841f68c5209525811b1cd487132edf922dabcbd261cad7d354d6b179229cd`) and `nrs-large.safetensors.sha256`. Made by our export script from the step-200,000 checkpoint; DataParallel prefix removed; round trip bit-exact; strict load into DLNR: ok. Weights fingerprint `0339296e6b2b8eaed86757550ca40ce04443c0121cef1c0435c177bee591fc65`. |

## Inputs and outputs

- **Inputs:** two rectified images, RGB, 0-255, any size. They are padded to a multiple of 32 (edge replication) and cropped back. Grayscale images work: `infer.py` reads them as three equal channels (our OAK-D Lite tests used grayscale).
- **Output:** the disparity of the left image in pixels (the network outputs flow; disparity = −flow).
- **Depth:** depth [m] = fx [px] x baseline [m] / disparity [px].
- **Precision:** FP32. The accuracy numbers below use IEEE FP32 (TF32 off). `infer.py` sets TF32 off on CUDA too.
- **Any stereo rig:** `rig_depth.py` rectifies a raw pair with the rig's calibration (an xclean `calib.json` or an OpenCV `cv2.stereoCalibrate` file; OpenCV's pinhole model), runs the model, and writes disparity and depth in the rectified left camera. Guide: `ANY_STEREO_RIG.md` in the code repository.
- **Runtime:** `infer.py` (one rectified pair) and `rig_depth.py` (raw pairs of a calibrated rig), PyTorch, CPU or CUDA. No ONNX or TensorRT export is part of this release.

**Terms used below.** *Pre-registered plan*: our plan, written before training (internal; available on request). *G-TEACH*: the release gate in that plan, with bars fixed before training. *T-W*: the training run that made these weights. *T0*: a separate run on TartanAir only, not this release. *P8*: the check that our evaluation harness reproduces the published RAFT-Stereo numbers.

## Accuracy

Our evaluation harness (not yet public; files fixed by sha256 at the G-TEACH run: `evaluate.py` 869f10c8…, `metrics.py` 88f0f3d0…, `protocol.py` 95f41eb2…, `datasets.py` f40c8af0…, `adapters.py` fa0789ff…). P8 (the RAFT-Stereo reproduction check) passed on 2026-11-05 with the earlier `evaluate.py`; the only later change is the IEEE FP32 default. Every row uses the WMG protocol: the per-set region WMG states (Middlebury 2014 non-occluded, all others all pixels) and the mean over images. Each benchmark was scored **once**, at the G-TEACH gate (2026-11-12). *Middlebury-H*: Middlebury 2014 training set at half resolution; *Booster-Q*: Booster training set at quarter resolution. The weights are the last checkpoint of a fixed-length run (step 200,000). No checkpoint was chosen by a benchmark or by the held-out monitor. Lower is better.

| Model | Middlebury-H bad-2 noc (%) | Middlebury 2021 bad-2 (%) | ETH3D bad-1 (%) | KITTI-12 bad-3 (%) | KITTI-15 bad-3 (%) | Booster-Q bad-2 (%) | Note |
|---|---|---|---|---|---|---|---|
| **nrs-large 0.1.0** (ours) | **3.91** | **8.44** | **2.04** | **3.38** | **4.59** | **9.22** | DLNR from random init on WMGStereo-150k only, 200k steps; IEEE FP32 |
| DLNR from scratch on WMGStereo-150k, published | 3.76 | 6.72 | 2.50 | 3.30 | 4.54 | 9.09 | WMG, arXiv 2504.16930v3, Table 2 (the same values are in Table 5); the row our run tries to reproduce. Their numbers, not scored by us. |
| RAFT-Stereo, public SceneFlow weights | 8.67 | 10.26 | 2.61 | 4.76 | 5.48 | 17.75 | our harness, P8 reference row, 2026-11-05; scored with TF32 convolutions (before the FP32 default) and not re-scored in FP32 |

- Image counts: Middlebury-H 15, Middlebury 2021 24, ETH3D 27, KITTI-12 194, KITTI-15 200, Booster-Q 228.
- **Against the published row:** 4 of 6 benchmarks agree within the P8 tolerance (±5 % or ±0.1). ETH3D is better than the published row (2.04 vs 2.50). Middlebury 2021 is worse (8.44 vs 6.72). We do not know the cause of the Middlebury 2021 gap. One possible cause is the data shortfall below (Training data); we have not tested that.
- **G-TEACH result: PASS (4 of 4 bars).** ETH3D 2.04 ≤ 3.37; Middlebury-H 3.91 ≤ 5.82; KITTI-15 4.59 ≤ 5.53; Booster-Q 9.22 ≤ 11.9. The bars were set before training in the pre-registered plan.
- **Held-out monitor (not a benchmark):** 200 TartanAir pairs from hold-out environments, which T-W never trained on. At step 200,000: EPE 1.65 px, bad-1 24.9 %, bad-3 11.4 %. This monitor did not select a checkpoint.
- **Real camera check (small, no ground truth):** 4 frames of one dim indoor scene from an OAK-D Lite (grayscale, 640x400). Where the camera's own on-board stereo gives depth (about 32 % of pixels), the median difference to nrs-large is 3.5 %, and about 80 % of those pixels agree within 10 %. This is agreement with another stereo method, not an accuracy measurement.

## Speed

Desktop only; there is no Jetson target for this model. Times are for the network forward pass of one pair, 32 iterations, FP32, after 3 warm-up pairs.

| Device | Precision | Input | Iterations | Time per pair | Label |
|---|---|---|---|---|---|
| NVIDIA RTX 5090 (PyTorch 2.9.1, CUDA 13.0) | FP32 (TF32 off) | 640x480 | 32 | 0.28 s (median of 20) | measured 2026-11-12 |
| CPU, 8 threads (Ryzen 9 9950X) | FP32 | 640x480 | 32 | 6.5 s (median of 5) | measured 2026-11-12 |

The first pair in a new process is slower on CUDA (kernel set-up: 0.9 s to 5.5 s in our runs).

## Training data

WMGStereo-150k is the only training data. Its Hugging Face dataset card carries a `bsd-3-clause` tag; the generator repository's BSD-3-Clause text is in `LICENSES/`. Full credits are in `ATTRIBUTION.md`.

| Source | Licence | How we know | Used in this release |
|---|---|---|---|
| WMGStereo-150k (indoor 71,800, flying 70,300, nature_filtered 13,669 = 155,769 pairs), Hugging Face revision 5ecc0dcd | BSD-3-Clause | **tag only** (Hugging Face card); generator repository BSD-3 text read | **yes (the only training data)** |
| TartanAir V1 (hold-out environments `seasonsforest_winter/Easy`, `carwelding`, `japanesealley`) | CC BY 4.0 | licence text read | **no training.** 200 hold-out pairs were used only as the monitor above. |

- **Data shortfall (stated in the pre-registered plan):** our `nature_filtered` set has 13,669 pairs. The WMG paper's nature split has 21,296 pairs (7,627 fewer, −36 %). We report this and did not correct it.
- **Never trained on:** TartanAir, Middlebury, ETH3D, KITTI, Booster, Spring, and any other group's model weights.
- **Recipe:** the authors' DLNR `train.sh` (batch 8, crop 384x736, 22 train iterations, 200k steps, AdamW 2e-4, OneCycle). WMG scene types are sampled in equal shares (WMG Sec. 4). The deviations are listed in the pre-registered plan: D1 bf16 autocast (FP32 correlation, upsampling and loss); D2 single GPU, micro-batch 2 with gradient accumulation 4; D3 held-out monitoring with no checkpoint selection; D4 per-sample seeding; D5 hold-out exclusion; D7 a micro-batch with no valid label pixel gets a zero loss (added at step 51,319; it fired 2 times in the run). D6 (float16 depth labels) does not apply to WMG.
- **Run:** seed 20261105; 200,000 steps; 92.0 GPU-hours of training steps on one RTX 5090 (about 3.8 GPU-days; the sum of logged s/step over the 2,000 logged 100-step blocks; 93.3 h with the steps redone after resumes); 2026-11-08 to 2026-11-12, with 9 resumes, all from the saved full state: a STOP-file stop at step 237, a non-finite-loss crash at step 51,411 (an all-empty-label micro-batch; fixed by D7), a DataLoader-worker crash after step 107,200, a manual PC restart (resume at 8,135), four power cuts (resumes at 110,689, 111,768, 117,505 and 179,183) and one Windows Update restart (resume at 141,698); run folder `TW_dlnr_wmg_s20261105`; final checkpoint `w_200000.pth`, sha256 `17b7e60be3c8e288dbbc3ef8789d76123e81462b3a1f308b1aae4c07e4a6d551`.

## Intended use

- Offline depth for maps and 3D reconstruction from rectified stereo.
- Labelling real stereo footage to train smaller models (our planned on-board model, `nrs-edge`, is not released yet).
- Research and benchmarking, with the numbers above.

**Out of scope:**
- real-time use on embedded boards (a smaller on-board model, `nrs-edge`, is planned);
- safety-critical use without an independent safety layer;
- unrectified input to `infer.py` (`rig_depth.py` rectifies first), fisheye lenses, and monocular input.

## Limitations

**What we do not claim:**
- No embedded (e.g. Jetson) timing is given; there is no real-device log.
- No robustness on glass or mirrors.
- No zero-shot result is claimed for a domain that was in the training data.
- No other group's model weights (DLNR or any other) were used. Only the DLNR authors' MIT code design was used, and it is credited.

**Known limits of this model:**
- **Refinement is all or nothing.** It runs only when every pixel of the 32nd update has a positive disparity (the authors' rule `disp.max() < 0` on the flow). One pixel at or below zero turns it off for the whole image. With these weights, refinement ran on 4 of 4 test pairs (CPU and CUDA). `infer.py` reports it for every pair.
- **CPU and GPU outputs are not identical.** On 4 test pairs (3 OAK-D Lite example pairs, 1 TartanAir hold-out pair) the median difference is 0.00001 px and the 99th percentile is below 0.006 px, but single pixels can differ by up to 0.66 px (the TartanAir pair; 0.006 px on the OAK pairs) (FP32 summation order, amplified over 32 iterations).
- **Synthetic-only training.** The model has seen no real camera noise, rolling shutter or real lens flare.
- **Slow:** 32 iterations of a 57 M-parameter network; not for on-board use.
- **Hard cases:** textureless and reflective surfaces, thin structures, low light. On Booster-Q (glass, mirrors, screens), the worst scenes score bad-2 means of TV 64 % (up to 83 % on one image), Mouthwash 41 %, Mirror 27 % and Washer 22 %.

## Citation and credits

The architecture and code are by Zhao et al. (High-Frequency Stereo Matching Network, CVPR 2023). Their code builds on RAFT-Stereo (Lipson, Teed, Deng, 3DV 2021). The training data is WMGStereo-150k (Yan, Raistrick, Deng, *What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?*, CVPR 2026, arXiv 2504.16930). Please cite all three. See `ATTRIBUTION.md` and `NOTICE`.

Contact: necessityrobotics@gmail.com
