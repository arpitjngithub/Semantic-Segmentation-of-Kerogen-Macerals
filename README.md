# Semi-Supervised Semantic Segmentation for Mineral Analysis

## Overview

This project focuses on segmenting minerals from micro rock samples using semi-supervised semantic segmentation techniques. Participating in the ThinkOnward Stranger Sections-2 competition, our team achieved a remarkable mIoU score of 0.49, placing 49th out of 295+ teams globally. This result represents a 115.2% improvement over previous results.

## Key Features

- **mIoU Score**: Achieved an impressive mean Intersection over Union (mIoU) score of 0.49.
- **Approach**: Utilized a combination of Vision Transformers (ViT), Autoencoder, and CNN-based approaches for image segmentation.
- **Techniques**: Applied transfer learning, data augmentation, and unsupervised learning to enhance model performance.
- **Ensembling**: Improved accuracy by 30% through ensembling techniques and training on patches rather than entire images.
- **Models**: Implemented and fine-tuned R2Attention-Unet and DeepLabv3+ models.
- **Experimentation**: Leveraged W&B track tool for meticulous experimentation and analysis, resulting in a 4-5% improvement.

## Installation

To get started with the project, ensure you have the following dependencies installed:

```bash
pip install torch torchvision tensorflow numpy pandas matplotlib scikit-learn wandb
```
### Data Preparation

Ensure your dataset is organized in the following structure:
```bash
/data /images /labels
```

### Training

Adjust configuration files to point to your dataset paths.

Run the training script:

```bash
python train.py --config config.yaml
```

### Evaluation
To evaluate the trained model, use:
```bash
python evaluate.py --checkpoint path/to/checkpoint.pth
```

## Models

- **Vision Transformers (ViT):** Utilized for capturing long-range dependencies in the images.
- **Autoencoder:** Employed to learn efficient representations of the input data.
- **CNN-based Approaches:** Applied for feature extraction and segmentation.
- **R2Attention-Unet:** Fine-tuned to capture detailed spatial information.
- **DeepLabv3+:** Optimized for accurate segmentation results.

## Results

- **Mean Intersection over Union (mIoU):** 0.49
- **Improvement:** 115.2% over previous results
- **Accuracy Gain:** Additional 30% through ensembling and patch-based training

## Experiment Tracking

Utilized **Weights & Biases (W&B)** for tracking experiments and analyzing model performance. Ensure you have a W&B account and API key to use this feature.

## Contributing

Feel free to contribute to this project by opening issues or submitting pull requests. Please contact arpitjain8302@gmail.com


## Acknowledgments

- Thanks to the ThinkOnward for providing the platform to showcase our work.
- Acknowledgment to the open-source community for the tools and libraries used in this project.

