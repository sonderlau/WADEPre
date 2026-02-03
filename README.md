# WADEPre: Wavelet-based Decomposition Model for Extreme Precipitation Nowcasting with Multi-Scale Learning

<div align="center">


<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&style=plastic" alt="PyTorch"></a><a href="https://lightning.ai/docs/pytorch/stable/"><img src="https://img.shields.io/badge/PyTorch_Lightning-2.4.0-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white&labelColor=181717&style=plastic" alt="PyTorch Lightning"></a><a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-00599C?style=for-the-badge&logo=open-source-initiative&logoColor=white&labelColor=181717&style=plastic" alt="License"></a>

<a href="https://kdd2026.kdd.org/ai4sciences-track-call-for-papers//"><img src="https://img.shields.io/badge/KDD_2026-Under_Review-b38808?style=for-the-badge&logo=acm&logoColor=white&labelColor=181717&style=plastic" alt="KDD 2026"></a> <a href="https://arxiv.org/abs/2602.02096"><img src="https://img.shields.io/badge/Arxiv-2602.02096-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=181717&style=plastic" alt="Arxiv"></a>



</div>



> Official Implementation of "**WADEPre**: **WA**velet-based **D**ecomposition Model for **E**xtreme **Pre**cipitation Nowcasting with Multi-Scale Learning"





*Authors*: Baitian Liu [1], Haiping Zhang [1], Huiling Yuan [2, 3], Dongjing Wang [1], Ying Li [4], Feng Chen [4], Hao Wu [1, *]



> 1. Department of Computer Science and Technology, Hangzhou Dianzi University, Hangzhou, Zhejiang  Province, China
> 2. State Key Laboratory of Severe Weather Meteorological Science and Technology, Nanjing University, Nanjing, China
> 3. Key Laboratory of Mesoscale Severe Weather, Ministry of Education, and School of Atmospheric Sciences, Nanjing University, Nanjing, China
> 4. Zhejiang Institute of Meteorological Sciences, Hangzhou, Zhejiang Province, China
> 
>*Corresponding author: Hao Wu



## 📢 News

- (🔥 New) [2026-02-03] Our paper is now available on arXiv.
- (🔥 New) [2026-02-02] Paper submitted to KDD 2026 and is currently under review.




<details>

<summary>History news</summary>

- [2026-01-23] Utility updated.
- [2025-11-17] Repository initiated.

</details>



---



## ⚡ Highlights
-  **Beyond Pixel-wise & Fourier**: Overcomes the *blurring* of MSE-based models and the *spatial leakage* of Fourier models via Discrete Wavelet Transform (DWT).
-  **Stable Optimization**: Implements a *dynamic weight annealing strategy* to prioritize structural learning before texture refinement, ensuring robust convergence for chaotic weather systems.
- **High-Fidelity Nowcasting**: Establishes new SOTA benchmarks on SEVIR and Shanghai Radar, delivering sharper images and superior CSI scores at extreme thresholds.



---





## 🏆  Results

We achieve state-of-the-art performance on the SEVIR and Shanghai Radar datasets.



TBD.



---



## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/sonderlau/WADEPre

cd WADEPre

# 2. Create environment
conda env create -f env.yaml
conda activate wadepre
```



---



## 📂 Data Preparation

### SEVIR Dataset

We use Vertically Integrated Liquid (VIL) mosaics in SEVIR for benchmarking precipitation nowcasting, predicting the future VIL up to 6\*10 minutes given 6\*10 minutes of context VIL, and resizing the spatial resolution to 128. The resolution is thus `6×128×128 → 6×128×128`.

We thank AWS for providing an online download service. Please download the SEVIR dataset from [AWS Open Data](https://registry.opendata.aws/sevir/). 

### Shanghai Radar Dataset

*Shanghai Radar*: The raw data spans a 460 × 460 grid covering a physical region of `460km × 398km`, with reflectivity values ranging from 0 to 70 dBZ. We resize the spatial resolution to 128. The resolution is thus `6×128×128 → 6×128×128`.

The Shanghai Radar dataset can be downloaded from the official [Zenodo repo](https://zenodo.org/records/7251972).



---



## 🚀 Usage

### Training

To train WADEPre on GPU(s):

```bash
# Change hyperparameters in the train.py
python train.py
```



### Evaluation

To evaluate the pre-trained model:

```bash
# Change settings in the eval.py
python eval.py
```



---



## 🤝 Acknowledgement <a href="https://www.nvidia.com/"><img src="https://img.shields.io/badge/Accelerated_by-NVIDIA_H100-76B900?style=for-the-badge&logo=nvidia&logoColor=white&labelColor=181717&style=plastic" alt="Hardware"></a>



Our implementation is heavily inspired by the following excellent works. We extend our thanks to the original authors.



Third-party libraries and tools:

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Muon](https://github.com/KellerJordan/Muon)
- [Draw.io](https://www.drawio.com/)



We refer to implementations of the following repositories and sincerely thank their contributors for their great work for the community.

- [Dilated ResNet](https://github.com/fyu/drn)
- [FPN](https://github.com/kuangliu/pytorch-fpn)
- [ConvLSTM](https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py)
- [MAU](https://github.com/ZhengChang467/MAU)
- [EarthFarseer](https://github.com/Alexander-wu/EarthFarseer)
- [SimVP](https://github.com/A4Bio/SimVP)
- [AlphaPre](https://github.com/linkenghong/AlphaPre)
