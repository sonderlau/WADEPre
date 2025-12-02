# WADE-Pre: Wavelet-based Disentanglement of Advection and Intensity for Mitigating the Smoothing Effect in Extreme Precipitation Nowcasting



By Baitian Liu [1], Haiping Zhang [1, 2], Luan Xu [3], Ying Li [3], Feng Chen [3], Qing Zhang [4], Xujian Fang [1, 2], Chonglei Zhu [2], Hao Wu [1, *****]



> 1. Department of Computer Science and Technology, Hangzhou Dianzi University, Hangzhou, Zhejiang  Province, China
>
> 2. Hangzhou Dianzi University Information Engineering College, Hangzhou, Zhejiang Province, China
>
> 3. Zhejiang Institute of Meteorological Sciences, Hangzhou, Zhejiang Province, China
>
> 4. Zhejiang Province Meteorological Bureau, Hangzhou, Zhejiang Province, China
>
> *****Corresponding author: Hao Wu





**WADE-Pre**: **WA**velet **D**isentanglement Model for Extreme **Pre**cipitation



**The source code will be released upon publication.**



> Last updated: Dec 2. 2025



## Introduction

**WADE-Pre** is a wavelet-based deep learning framework designed to tackle the <u>smoothing effect</u> in extreme precipitation nowcasting. By explicitly disentangling radar imagery into `stable large-scale advection` (low-frequency) and `volatile local intensity` (high-frequency) components, the model overcomes the regression-to-the-mean dilemma inherent in standard pixel-wise optimization. Powered by a `multi-task curriculum learning strategy`, WADE-Pre achieves state-of-the-art performance on the SEVIR benchmark, significantly improving forecast accuracy and structural fidelity for high-impact weather events compared to Fourier-based and deterministic baselines.

This repository contains the part of the training and inference code for running WADE-Pre to make predictions (6 --> 6) on SEVIR VIL datasets.



## SEVIR Dataset

**S**torm **EV**ent **I**mage**R**y (SEVIR) dataset is a spatiotemporally aligned dataset containing over 10,000 weather events. We adopt NEXRAD Vertically Integrated Liquid (VIL) mosaics in SEVIR for benchmarking precipitation nowcasting, i.e., to predict the future VIL up to 6\*10 minutes given 6\*10 minutes context VIL, and resize the spatial size to 128. The resolution is thus `6×128×128 → 6×128×128`.

We thank AWS for providing online download service, for more details please refer to [AWS - Storm EVent ImageRy (SEVIR)](https://registry.opendata.aws/sevir/)



## Code



### Environment

```bash
conda env create -f env.yaml
conda activate stormwave
```





### Evaluation

Open the `eval.py` file, and replace the weight file path with your path. Then run the command:

```bash
python eval.py 
```



### Training

```python
python train.py
```



When you start training, these folders may offer you useful information:

- `logs`  All the metrics during training have been recorded in this file, including hyper-parameters.
- `checkpoints` Mode weight file.



## Citation

```
@article{liuWADEPre2025,
  author = {Baitian, Liu and Haiping, Zhang and Xujian, Fang},
  title = {{WADE-Pre}: Wavelet-based Disentanglement of Advection and Intensity for Mitigating the Smoothing Effect in Extreme Precipitation Nowcasting},
  journal = {Geophysical Research Letters},
  volume = {},
  number = {},
  pages = {},
  doi = {},
  url = {},
  year = {2025}
}

```



## Credit

Our implementation is heavily inspired by the following excellent works. We extend our thanks to the original authors.



Third-party libraries and tools:

- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [Muon](https://github.com/KellerJordan/Muon)
- [Draw.io](https://www.drawio.com/)



We refer to implementations of the following repositories and sincerely thank their contribution for the community.

- [U-Net](https://github.com/himashi92/vanila-unet/blob/master/model/Unet.py)
- [ConvLSTM](https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py)
- [EarthFarseer](https://github.com/Alexander-wu/EarthFarseer)
- [SimVP](https://github.com/A4Bio/SimVP)
- [AlphaPre](https://github.com/linkenghong/AlphaPre)
- [FACL](https://github.com/argenycw/FACL) - MIT License