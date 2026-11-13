# Attribution: nrs-large 0.1.0

This file credits every work `nrs-large` 0.1.0 is built from. The exact licence texts are in `LICENSES/`; the short legal notice is `NOTICE`.

**Licence of the nrs-large 0.1.0 weights and scripts: Apache License, Version 2.0 (`LICENSE`).**

## 1. Code and network design

The network was trained by Necessity Robotics from random initialisation. Only the authors' MIT-licensed code design is used; **no DLNR weights** were used for training.

| Work | Used for | Licence | Copyright | Please cite |
|---|---|---|---|---|
| DLNR / High-frequency-Stereo-Matching-Network, github.com/David-Zhao-1997/High-frequency-Stereo-Matching-Network, commit c10475f | `nrs-large` network | MIT (`LICENSES/DLNR_MIT.txt`) | (c) 2021 Princeton Vision & Learning Lab (the RAFT-Stereo licence the repository carries) | Zhao et al., *High-Frequency Stereo Matching Network*, CVPR 2023; Lipson, Teed, Deng, *RAFT-Stereo*, 3DV 2021 |

## 2. Training data

The only training data is WMGStereo-150k. Its licence statement is a tag on the dataset card (see the table).

| Source | Licence | Evidence | Credit and citation | Used |
|---|---|---|---|---|
| WMGStereo-150k (indoor, nature_filtered, flying), Hugging Face revision 5ecc0dcd | BSD-3-Clause | `bsd-3-clause` tag on the Hugging Face dataset card (present since the first author upload); the generator repository carries the BSD-3-Clause text (`LICENSES/WMGStereo_generator_BSD-3-Clause.txt`). If the dataset licence changes, we will stop distributing these weights and retrain without it. | Yan, Raistrick, Deng, *What Makes Good Synthetic Training Data for Zero-Shot Stereo Matching?*, CVPR 2026, arXiv 2504.16930. Notice: Copyright (c) 2025, Princeton Vision & Learning Lab, with the three conditions and the disclaimer | **yes, the only training data** (155,769 pairs) |

**Not training data:**
- **Held-out monitor only:** 200 TartanAir V1 pairs from the hold-out environments (CC BY 4.0; Wang et al., *TartanAir*, IROS 2020; CMU AirLab). They scored the run's progress and selected nothing.
- **Evaluation only, never trained on:** Middlebury, ETH3D, KITTI 2012/2015, Booster.
