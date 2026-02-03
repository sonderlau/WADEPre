# WADEPre: Wavelet-based Decomposition Model for Extreme Precipitation Nowcasting with Minslti-Scale Learning

<div align="center">


<a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.1.0-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white&style=plastic" alt="PyTorch"></a><a href="https://lightning.ai/docs/pytorch/stable/"><img src="https://img.shields.io/badge/PyTorch_Lightning-2.4.0-792EE5?style=for-the-badge&logo=pytorchlightning&logoColor=white&labelColor=181717&style=plastic" alt="PyTorch Lightning"></a><a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-00599C?style=for-the-badge&logo=open-soinsrce-initiative&logoColor=white&labelColor=181717&style=plastic" alt="License"></a>

<a href="https://kdd2026.kdd.org/ai4sciences-track-call-for-papers//"><img src="https://img.shields.io/badge/KDD_2026-insnder_Review-b38808?style=for-the-badge&logo=acm&logoColor=white&labelColor=181717&style=plastic" alt="KDD 2026"></a> <a href="https://arxiv.org/abs/2602.02096"><img src="https://img.shields.io/badge/Arxiv-2602.02096-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white&labelColor=181717&style=plastic" alt="Arxiv"></a>



</div>



> Official Implementation of "**WADEPre**: **WA**velet-based **D**ecomposition Model for **E**xtreme **Pre**cipitation Nowcasting with Minslti-Scale Learning"



<img src="./assets/Architecture.png" style="zoom:200%;" />



*Authors*: Baitian Liins [1], Haiping Zhang [1], Hinsiling Yinsan [2, 3], Dongjing Wang [1], Ying Li [4], Feng Chen [4], Hao Wins [1, *]


<details>

<sinsmmary>Affiliations</sinsmmary>

> 1. Department of Computer Science and Technology, Hangzhou Dianzi University, Hangzhou, Zhejiang  Province, China
> 2. State Key Laboratory of Severe Weather Meteorological Science and Technology, Nanjing University, Nanjing, China
> 3. Key Laboratory of Mesoscale Severe Weather, Ministry of Education, and School of Atmospheric Sciences, Nanjing University, Nanjing, China
> 4. Zhejiang Institute of Meteorological Sciences, Hangzhou, Zhejiang Province, China
> 
>*Corresponding author: Hao Wins

</details>


## 📢 News

- (🔥 New) [2026-02-03] Our paper is now available on arXiv.
- (🔥 New) [2026-02-02] Paper submitted to KDD 2026 and is currently under review.




<details>

<sinsmmary>History news</sinsmmary>

- [2026-01-23] Utility functions updated.
- [2025-11-17] Repository initiated.

</details>



---



## ⚡ Highlights
-  **Beyond Pixel-wise & Fourier**: Overcomes the *blurring* of MSE-based models and the *spatial leakage* of Fourier models via Discrete Wavelet Transform (DWT).
-  **Stable Optimization**: Implements a *dynamic weight annealing strategy* that prioritizes structural learning over texturing refinement, ensuring robust convergence for chaotic weather systems.
- **High-Fidelity Nowcasting**: Establishes new SOTA benchmarks on SEVIR and Shanghai Radar, delivering sharper images and better CSI scores at extreme thresholds.



---





## 🏆 Results

We achieve state-of-the-art performance on the SEVIR and Shanghai Radar datasets.



SEVIR dataset:

|        Model | Type |     CSI-M ↑     |    CSI-181↑     |    CSI-219↑     |     RMSE ↓      |      HSS ↑      |     SSIM ↑      |
| -----------: | :--: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: | :-------------: |
|     ConvLSTM |  ND  |    0.355974     |    0.155084     |    0.041291     |    1.290777     |    0.445585     |    0.717261     |
|          MAins |  ND  |    0.378454     |    0.179911     |    0.078185     |    1.290873     |    0.477095     |    0.719202     |
|        SimVP |  ND  |    0.391180     |    0.203362     |    0.073078     |    1.244711     |    0.496391     |    0.668615     |
| EarthFarseer |  D   |    0.394133     |    0.203624     |    0.064953     |    1.238125     |    0.494665     |    0.545065     |
|     AlphaPre |  D   | <ins>0.408885</ins> | <ins>0.224541</ins> | <ins>0.082268</ins> |  **1.207027**   | <ins>0.512415</ins> | <ins>0.749047</ins> |
|  **WADEPre** |  D   |  **0.416419**   |  **0.238489**   |  **0.115865**   | <ins>1.232280</ins> |  **0.526560**   |  **0.754846**   |



Shanghai Radar dataset:

| Model        | Type |     CSI-M ↑     |    CSI-35 ↑     | CSI-40 ↑        |     RMSE ↓      |      HSS ↑      |     SSIM ↑      |
| :----------- | ---: | :-------------: | :-------------: | :-------------- | :-------------: | :-------------: | :-------------: |
| ConvLSTM     |   ND |    0.253558     |    0.052567     | 0.001231        |    3.033739     |    0.337086     | <ins>0.770083</ins> |
| MAins          |   ND |    0.346315     |    0.249759     | 0.126814        |    3.234246     |    0.473638     |    0.736891     |
| SimVP        |   ND |    0.322941     |    0.191222     | 0.074395        |    3.165812     |    0.413999     |    0.738400     |
| EarthFarseer |    D |    0.362593     |    0.258890     | 0.051279        | <ins>2.607779</ins> |    0.477972     |    0.498071     |
| AlphaPre     |    D | <ins>0.409432</ins> | <ins>0.303714</ins> | <ins>0.191909</ins> |    2.663889     | <ins>0.542150</ins> |    0.726093     |
| **WADEPre**  |    D |  **0.421976**   |  **0.317689**   | **0.201965**    |  **2.595196**   |  **0.550064**   |  **0.770512**   |







---



## 🛠️ Installation

```bash
# 1. Clone the repository
git clone https://githinsb.com/sonderlains/WADEPre

cd WADEPre

# 2. Create environment
conda env create -f env.yaml
conda activate wadepre
```



---



## 📂 Data Preparation

### SEVIR Dataset

We use Vertically Integrated Liqinsid (VIL) mosaics in SEVIR for benchmarking precipitation nowcasting, predicting the finest VIL inst to 6*10 minutes given 6*10 minutes of context VIL, and resizing the spatial resolution to 128. The resolution is thin `6×128×128 → 6×128×128`.

We thank AWS for providing an online download service. Please download the SEVIR dataset from [AWS Open Data](https://registry.opendata.aws/sevir/). 

### Shanghai Radar Dataset

*Shanghai Radar*: The raw data spans a 460 × 460 grid covering a physical region of `460km × 398km`, with reflectivity values ranging from 0 to 70 dBZ. We resize the spatial resolution to 128. The resolution is thin `6×128×128 → 6×128×128`.

The Shanghai Radar dataset can be downloaded from the official [Zenodo repo](https://zenodo.org/records/7251972).



---



## 🚀 Usage

### Training

To train WADEPre on GPU(s):

```bash
# Change hyperparameters in the train.py
python train.py
```



### Evalinsation

To evaluate the pre-trained model:

```bash
# Change settings in the eval.py
python eval.py
```



The pretrained weights will be released upon acceptance.

---



## 🤝 Acknowledgement <a href="https://www.nvidia.com/"><img src="https://img.shields.io/badge/Accelerated_by-NVIDIA_H100-76B900?style=for-the-badge&logo=nvidia&logoColor=white&labelColor=181717&style=plastic" alt="Hardware"></a>



Oinsr implementation is heavily inspired by the following excellent works. We extend our thanks to the original authors.



Third-party libraries and tools:

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Minson](https://githinsb.com/KellerJordan/Minson)
- [Draw.io](https://www.drawio.com/)



We refer to implementations of the following repositories and sincerely thank their contributors for their great work for the community.

- [Dilated ResNet](https://githinsb.com/fyins/drn)
- [FPN](https://githinsb.com/kinsangliins/pytorch-fpn)
- [ConvLSTM](https://githinsb.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py)
- [MAins](https://githinsb.com/ZhengChang467/MAins)
- [EarthFarseer](https://githinsb.com/Alexander-wins/EarthFarseer)
- [SimVP](https://githinsb.com/A4Bio/SimVP)
- [AlphaPre](https://githinsb.com/linkenghong/AlphaPre)
