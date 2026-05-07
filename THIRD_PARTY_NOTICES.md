# Third-Party Notices

This file records source attribution for code ideas, adapted
implementations, and dataset sources used by the benchmark. It does not replace
local source comments or docstrings near adapted code. The repository license is
Apache-2.0. The notices below preserve upstream terms for adapted third-party
source material and do not change external dataset terms.

## Code And Algorithms

### Time-Series-Library

- Used by: `models/dlinear.py`, `models/components/attention.py`,
  `models/components/embedding.py`, `models/components/encoder_decoder.py`,
  `models/components/masking.py`, and `models/patchtst.py`.
- Reference: Zeng et al., 2023.
- Paper: <https://arxiv.org/abs/2205.13504>
- Reference: Nie et al., 2023.
- Paper: <https://openreview.net/forum?id=Jbdc0vTOcol>
- Repo: <https://github.com/thuml/Time-Series-Library>
- Upstream license: MIT License.
- Upstream copyright: Copyright (c) 2021 THUML @ Tsinghua University.
- Required MIT notice:

```text
MIT License

Copyright (c) 2021 THUML @ Tsinghua University

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### TSMixer

- Used by: `models/tsmixer.py`.
- Reference: Chen et al., 2023.
- Paper: <https://arxiv.org/abs/2303.06053>
- Repo: <https://github.com/google-research/google-research/tree/master/tsmixer>
- Source file: upstream Google Research `tsmixer_basic/models/tsmixer.py`
  implementation:
  <https://github.com/google-research/google-research/blob/master/tsmixer/tsmixer_basic/models/tsmixer.py>.
- Upstream copyright: Copyright 2026 The Google Research Authors.
- Upstream license: Apache-2.0.
- Modification notice: translated from TensorFlow/Keras into the repository's
  PyTorch Lightning style.

### Chronos-2 Forecasting

- Used by: `models/chronos2.py` and `models/pretrained_loader.py`.
- Project: Chronos Forecasting.
- Reference: Ansari et al., 2025.
- Paper: <https://arxiv.org/abs/2510.15821>
- Repo: <https://github.com/amazon-science/chronos-forecasting>
- Model source: <https://huggingface.co/amazon/chronos-2>
- Upstream package: `chronos-forecasting==2.2.2`.
- Upstream license: Apache-2.0.
- Scope: package-level Chronos-2 pretrained-model loading. Tested runs log a
  local pretrained snapshot artifact for reconstruction.

### Randomized Smoothing For Regression

- Used by: `models/randomized_smoothing.py`.
- Reference: Miri Rekavandi et al., 2024.
- Paper: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/f21a76d688be0553c943a6b6c1d4bb1f-Abstract-Conference.html>
- Repo: <https://github.com/arekavandi/Certified_adv_RRegression>
- Upstream repository license: no explicit repository license file found.
- Scope: paper-level algorithm reference.

### PGD Adversarial Training

- Used by: `models/components/adversarial.py`,
  `models/base_module.py`, and `configs/pipelines/adversarial_training.yaml`.
- Reference: Madry et al., 2018.
- Paper: <https://openreview.net/forum?id=rJzIBfZAb>
- Repo: <https://github.com/MadryLab/mnist_challenge>
- Upstream repository license: MIT License.
- Scope: paper-level PGD adversarial-training reference.

### Reversible Instance Normalization

- Used by: `models/components/revin.py`, `models/base_module.py`, and
  `configs/pipelines/revin.yaml`.
- Reference: Kim et al., 2021.
- Paper: <https://openreview.net/forum?id=cGDAkQo1C0p>
- Repo: <https://github.com/ts-kim/RevIN>
- Upstream repository license: MIT License.
- Scope: local RevIN layer adapted to the repository tensor layout and target
  projection semantics.

### ModernTCN

- Used by: `models/moderntcn.py`.
- Reference: Luo and Wang, 2024.
- Paper: <https://openreview.net/forum?id=vpJMJerXHU>
- Repo: <https://github.com/luodhhh/ModernTCN>
- Upstream repository license: MIT License.
- Scope: compact long-term-forecasting implementation adapted to the
  repository's PyTorch Lightning style.

### Adaptive Robust Loss

- Used by: `metrics/adaptive_robust_loss.py`.
- Reference: Barron, 2019.
- Paper: <https://arxiv.org/abs/1701.03077>
- Repo: <https://github.com/jonbarron/robust_loss_pytorch>
- Upstream copyright: Copyright 2019 The Google Research Authors.
- Upstream license: Apache-2.0.
- Modification notice: adapted into the repository's loss interface and
  PyTorch tensor style, including the spline resource used for the partition
  function approximation.

### Robustness Metric Comparisons

- Used by: comparison diagnostics in `metrics/reference_normalized.py`
  and `testing/meta_analysis.py`.
- Scope: forecasting adaptations of mCE-style reference-normalized diagnostics,
  mPC-style and rPC-style mean perturbed performance diagnostics, and
  Taori-style effective robustness residuals. Plotting code is repository-owned.
- Reference: Hendrycks and Dietterich, 2019.
- Paper: <https://openreview.net/forum?id=HJz6tiCqYm>
- Repo: <https://github.com/hendrycks/robustness/tree/master/ImageNet-C>
- Upstream repository license: Apache-2.0.
- Reference: Michaelis et al., 2019.
- Paper: <https://arxiv.org/abs/1907.07484>
- Repo: <https://github.com/bethgelab/robust-detection-benchmark>
- Upstream repository license: MIT License.
- Reference: Taori et al., 2020.
- Paper: <https://proceedings.neurips.cc/paper/2020/hash/d8330f857a17c53d217014ee776bfd50-Abstract.html>
- Repo: <https://github.com/modestyachts/imagenet-testbed>
- Upstream repository license: MIT License.

### Randomized Training Noise Augmentation

- Used by: `configs/pipelines/randomized_training.yaml`, `data/dataset.py`,
  and `utils/noise.py`.
- Reference: Yoon et al., 2022.
- Paper: <https://proceedings.mlr.press/v151/yoon22a.html>
- Repo: <https://github.com/tetrzim/robust-probabilistic-forecasting>
- Upstream repository license: no explicit repository license file found.
- Scope: paper-level randomized-training and noise-augmentation pattern
  reference.

### Sensor Failure-Mode Taxonomy

- Used by: `data/perturbations.py`.
- Topic: Sensor Failure Modes.
- Reference: Brandt et al., 2025.
- Paper: <https://openreview.net/forum?id=9aElHWiZ72>
- Repo: <https://github.com/JBrandt97/FaultsToFeatures>
- Upstream repository license: Creative Commons Attribution-NonCommercial 4.0
  International (CC BY-NC 4.0).
- Scope: selected perturbation operator semantics only. The
  `fault_augmentation` training recipe, scheduling, and benchmark method are
  repository-owned.

## Dataset Sources

Dataset acquisition and terms are documented in `data/README.md`. The benchmark
uses external sources for `ETTh1`, `traffic`, `BeijingAir_Tiantan`, and
`Penmanshiel_Hourly_WT08`. Raw and processed datasets are not redistributed by
default.

- `ETTh1`: ETT benchmark data archive:
  <https://github.com/zhouhaoyi/ETDataset>. Upstream repository license:
  Creative Commons Attribution-NoDerivatives 4.0 International
  (CC BY-ND 4.0).
- `traffic`: standard long-term-forecasting `traffic/traffic.csv` layout from
  the THUML Time-Series-Library dataset collection:
  <https://huggingface.co/datasets/thuml/Time-Series-Library/blob/main/traffic/traffic.csv>.
  Upstream dataset license: Creative Commons Attribution 4.0 International
  (CC BY 4.0). This is the exact THUML Traffic CSV used by this benchmark.
  Traffic Hourly provenance is associated with the Monash time-series
  forecasting archive:
  <https://doi.org/10.5281/zenodo.4656132>, also licensed under CC BY 4.0.
  Underlying PeMS source documentation:
  <https://dot.ca.gov/programs/traffic-operations/mpr/pems-source>.
- `BeijingAir_Tiantan`: Beijing Multi-Site Air Quality source:
  <https://doi.org/10.24432/C5RK5G>.
  Upstream dataset license: Creative Commons Attribution 4.0 International
  (CC BY 4.0).
- `Penmanshiel_Hourly_WT08`: Penmanshiel wind-farm SCADA source:
  <https://zenodo.org/records/16807304>, DOI
  <https://doi.org/10.5281/zenodo.16807304>. The benchmark uses the v3
  record's WT01-10 SCADA ZIP files for 2016 through 2022, because WT08 is in
  that file group. Later 2023 and 2024 files in v3 are not part of the frozen
  benchmark slice. Upstream dataset license: Creative Commons Attribution 4.0
  International (CC BY 4.0).
