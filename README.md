# LPNSR: Prior-Enhanced Diffusion Image Super-Resolution via LR-Guided Noise Prediction

A diffusion-based image super-resolution method that learns to predict optimal noise maps for efficient sampling.

---

> This project presents a novel approach to image super-resolution by training a noise predictor that estimates optimal noise maps for the diffusion process. The noise predictor enables initializing the sampling process at an intermediate state, significantly reducing the number of required sampling steps while maintaining high-quality results.

---

## Features

- **Efficient Sampling**: Only 4 sampling steps required for high-quality super-resolution
- **Noise Predictor**: Learn to predict optimal noise maps for partial diffusion initialization
- **Real-world SR**: Handles complex real-world degradations
- **SwinIR Integration**: Optional SwinIR refinement for enhanced details

## Visual Results

<div align="center">
  <b>4× Real-world Super-Resolution</b>
  <br><br>

  [![Demo 1](https://imgsli.com/i/YOUR_IMAGE_ID.jpg)](https://imgsli.com/YOUR_LINK_1)
  &nbsp;&nbsp;
  [![Demo 2](https://imgsli.com/i/YOUR_IMAGE_ID.jpg)](https://imgsli.com/YOUR_LINK_2)
  &nbsp;&nbsp;
  [![Demo 3](https://imgsli.com/i/YOUR_IMAGE_ID.jpg)](https://imgsli.com/YOUR_LINK_3)

</div>

## Requirements

- Python 3.10, PyTorch 2.9.1+cu130, Xformers 0.0.33.post2
- A suitable conda environment named `lpnsr` can be created and activated with:

```bash
conda create -n lpnsr python=3.10
conda activate lpnsr

# Install PyTorch with CUDA support 
#CUDA 13.0
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130

# Install other dependencies
pip install -r requirements.txt

# Install xformers for acceleration
pip install xformers==0.0.33.post2
```

## Pre-trained Models

Download all pre-trained models from [腾讯微云](https://share.weiyun.com/wbhPDZKw) and place them in the `pretrained/` folder:

| Model | Description |
|-------|-------------|
| `autoencoder_vq_f4.pth` | VQGAN encoder/decoder (4x spatial compression) |
| `resshift_realsrx4_s4_v3.pth` | Pre-trained ResShift UNet |
| `noise_predictor.pth` | Trained noise predictor |
| `003_realSR_BSRGAN_DFOWMFC_s64w8_SwinIR-L_x4_GAN.pth` | SwinIR for refinement |

## Quick Start

### :rocket: Inference

```bash
python LPNSR/inference.py -i [image folder/image path] -o [output folder]
```

### :test_tube: Testing

```bash
python LPNSR/test.py --lq [lq image folder] --gt [gt image folder]
```

### :railway_car: Online Demo

Launch the Gradio demo:
```bash
python LPNSR/app.py
```

Then open `http://127.0.0.1:7860` in your browser.

## Training

### :turtle: Preparing Stage

1. Prepare training data in `traindata/` folder (high-resolution images)
2. Download the pre-trained models (see above)
3. Adjust the configuration in `configs/train_noise_predictor.yaml`

### :dolphin: Begin Training

```bash
python LPNSR/train_noise_predictor.py --config LPNSR/configs/train_noise_predictor.yaml
```

### :whale: Resume from Interruption

```bash
python train_noise_predictor.py --config LPNSRconfigs/train_noise_predictor.yaml --resume LPNSR/experiments/noise_predictor/checkpoints/check_point_xx.pth
```

## Method Details

### Architecture

The method consists of three main components:

1. **VQGAN Encoder/Decoder**: 4x spatial compression for efficient latent space processing
2. **Noise Predictor**: Predicts optimal noise maps for partial diffusion initialization
3. **ResShift UNet**: Diffusion model for latent space super-resolution

### Diffusion Process

- **Forward Process**: Adds residual between HR and LR to construct intermediate states
- **Noise Schedule**: Exponential schedule with flexible control
- **Sampling**: Starts from predicted intermediate state, reducing required steps

## Acknowledgement

This project is based on:
- [ResShift](https://github.com/zsyOAOA/ResShift) - Efficient diffusion model for image SR
- [BasicSR](https://github.com/XPixelGroup/BasicSR) - Basic super-resolution toolbox
- [SwinIR](https://github.com/JingyunLiang/SwinIR) - Swin Transformer for image restoration
- [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) - Degradation simulation

## License

This project is licensed under the MIT License.

## Contact

If you have any questions, please feel free to open an issue or contact the maintainer.
