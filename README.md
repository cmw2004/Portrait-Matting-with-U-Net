# Portrait Matting with U-Net

High-precision portrait matting project based on deep learning, supporting background replacement and transparent background generation.

## ✨ Features

- 🎯 **High Precision Matting**: UNet + ResNet18 + Attention mechanism
- 🖼️ **Background Replacement**: One-click replace any background
- 🔍 **Edge Optimization**: Gradient loss + Laplacian edge loss for sharper edges
- 📊 **Data Augmentation**: Rich augmentation strategies to improve generalization
- ⚡ **Fast Inference**: GPU acceleration support

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirement.txt
```

### 2. Prepare Dataset

Organize your data in the following structure:

```
data/portrait/
├── images/        # Original RGB images
└── masks/         # Corresponding alpha mattes (grayscale, range 0-1)
```

### 3. Train Model

```bash
# Basic training (30 epochs)
python train.py --epochs 30

# Optimized training (50 epochs, better edge quality)
python train.py --epochs 50

# Custom parameters
python train.py --epochs 50 --batch 16 --lr 1e-4 --size 320
```

**Training Parameters:**
- `--data_root`: Dataset path (default: `./data/portrait/`)
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 8)
- `--lr`: Learning rate (default: 1e-4)
- `--size`: Input image size (default: 320)
- `--ckpt_dir`: Checkpoint save path (default: `./checkpoints/`)

### 4. Inference - Matting

```bash
# Basic matting
python infer.py --model ./checkpoints/best.pth --img your_image.jpg --out ./output/

# Batch processing
python infer.py --model ./checkpoints/best.pth --img folder/*.jpg --out ./output/
```

**Output Files:**
- `mask.png` - Grayscale mask (foreground white, background black)
- `cutout.png` - Transparent background cutout (RGBA format, ready to use)
- `composite.png` - Black background composite preview

### 5. Background Replacement

```bash
# Replace background
python replace_bg.py --model ./checkpoints/best.pth \
                     --img your_photo.jpg \
                     --background new_background.jpg \
                     --out ./replaced_output/

# Add edge smoothing
python replace_bg.py --model ./checkpoints/best.pth \
                     --img your_photo.jpg \
                     --background new_background.jpg \
                     --smooth 1.5 \
                     --out ./replaced_output/
```

**Parameters:**
- `--img`: Image to replace background
- `--background`: New background image
- `--smooth`: Edge smoothing factor (0=no smoothing, 1-3=light smoothing)

## 📂 Project Structure

```
.
├── model.py              # UNet model definition (ResNet18 encoder + Attention)
├── dataset.py            # Dataset loading and augmentation
├── losses.py             # Loss functions (BCE + L1 + Edge + Gradient)
├── train.py              # Training script
├── infer.py              # Inference script (matting)
├── replace_bg.py         # Background replacement script
├── utils.py              # Utility functions (save checkpoint, etc.)
├── requirement.txt       # Dependencies list
├── README.md             # Project documentation
└── checkpoints/          # Model checkpoint save directory
```

## 🔧 Model Architecture

### Core Components

1. **Encoder**: ResNet18 (pretrained) - Extract multi-scale features
2. **Decoder**: Upsampling + Skip connections - Restore spatial resolution
3. **Attention Mechanism**: Attention Gates - Focus on foreground regions
4. **Activation Function**: Sigmoid - Output alpha values in [0,1]

### Loss Function

Composite Loss = BCE + 10×L1 + 15×Edge + 20×Gradient

- **BCE Loss**: Binary classification base loss
- **L1 Loss**: Overall accuracy optimization
- **Edge Loss**: Laplacian edge loss
- **Gradient Loss**: Sobel gradient loss (newly added, focuses on edge details)

### Data Augmentation

- Horizontal Flip (p=0.5)
- Random Brightness Contrast (p=0.5)
- Color Jitter (p=0.5)
- Gaussian Noise (p=0.3)
- Random Blur (p=0.2)
- Gamma Transform (p=0.3)
- HSV Adjustment (p=0.3)

### Training Strategy

- **Optimizer**: Adam (lr=1e-4)
- **Learning Rate Schedule**: MultiStepLR (lr×0.1 at epoch 20 and 40)
- **Batch Size**: 8 (adjustable based on GPU memory)
- **Input Size**: 320×320

## 📊 Performance Metrics

Performance on validation set:
- **MAE**: ~0.014 (Mean Absolute Error)
- **Training Epochs**: 50 epochs
- **Training Speed**: ~10 it/s (single RTX GPU)

## 🎨 Usage Examples

### Example 1: ID Photo Background Replacement

```bash
# Replace ID photo background with solid color
python replace_bg.py --model ./checkpoints/best.pth \
                     --img id_photo.jpg \
                     --background blue_bg.jpg \
                     --out ./id_output/
```

### Example 2: Product Image Matting

```bash
# Generate transparent background product image
python infer.py --model ./checkpoints/best.pth \
                --img product.jpg \
                --out ./product_output/
```

## 🛠️ Advanced Usage

### Resume Training from Checkpoint

Modify `train.py` to add resume functionality:

```python
# Load previous checkpoint to continue training
checkpoint = torch.load('./checkpoints/best.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['opt'])
start_epoch = checkpoint['epoch'] + 1
```

### Adjust Loss Function Weights

Modify weights in `losses.py`:

```python
def composite_loss(pred, target):
    l_bce = bce(pred, target)
    l_l1 = F.l1_loss(pred, target)
    l_edge = laplacian_edge(pred - target)
    l_grad = gradient_loss(pred, target)
    
    # Custom weights
    return l_bce + 10 * l_l1 + 15 * l_edge + 20 * l_grad
```

## 📝 Notes

1. **NumPy Version**: Requires NumPy < 2.0 (compatibility issue)
   ```bash
   pip install "numpy<2"
   ```

2. **GPU Memory Requirements**: 
   - Training: At least 4GB VRAM (batch_size=8)
   - Inference: ~2GB VRAM

3. **Data Format**:
   - Input images: RGB, any size
   - Mask: Grayscale, value range [0, 1], same size as image

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- UNet: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Attention Gates: [Attention U-Net](https://arxiv.org/abs/1804.03999)

## 📧 Contact

For questions, please submit a [GitHub Issue](https://github.com/cmw2004/Portrait-Matting-with-U-Net/issues)
