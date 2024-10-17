# WaveRoRa: Wavelet Rotary Router Attention for Multivariate Time Series Forecasting

### This is the official implementation of **WaveRoRa: Wavelet Rotary Router Attention for Multivariate Time Series Forecasting**.

![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg?style=plastic)  ![PyTorch 2.3.0](https://img.shields.io/badge/Pytorch-2.3.0(+cu118)-da282a?style=plastic)  ![numpy 1.24.1](https://img.shields.io/badge/numpy-1.24.1-2ad82a?style=plastic)  ![pandas 2.0.3](https://img.shields.io/badge/pandas-2.0.3-39a8da?style=plastic)  ![optuna 3.6.1](https://img.shields.io/badge/optuna-3.6.1-a398da?style=plastic)  ![einops 0.7.0](https://img.shields.io/badge/einops-0.7.0-a938da?style=plastic)


🚩**News**(Oct 17, 2024): We upload the code to Github. The repo is currently private.

# Key Designs of the proposed WaveRoRA🔑
🤠 We propose a deep architecture to process time series data in Wavelet domain. We decompose time series into multi-scale wavelet coefficients through Discrete Wavelet Transform (DWT) and use deep models to capture intra- and inter-series dependencies.

🤠 We propose a novel Rotary Router Attention (RoRA) mechanism. Compared to vanilla Softmax Attention, RoRA utilizes rotary position embedding (RoPE) to model the relative position information between different sequence elements. In addition, RoRA introduces a fixed number of router tokens $a$ to aggregate information from $KV$ matrices and reassign it to $Q$ matrix. RoRA achieves a good balance between computational efficiency and the ability to capture global dependencies.

🤠 We conduct extensive experiments and find that transfering other deep model architectures to wavelet domain also leads to better predicting results.

# Results✅
# Getting Start🛫
# Datasets🔗
# Acknowledgements🙏
# Citation🙂