# Portrait Matting - 人像抠图

基于深度学习的高精度人像抠图项目,支持背景替换和透明背景生成。

## ✨ 特性

- 🎯 **高精度抠图**: 使用UNet + ResNet18 + Attention机制
- 🖼️ **背景替换**: 一键替换任意背景
- 🔍 **边缘优化**: 梯度损失 + Laplacian边缘损失,边缘更清晰
- 📊 **数据增强**: 丰富的数据增强策略提高泛化能力
- ⚡ **快速推理**: 支持GPU加速

## 📋 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA (可选,GPU加速)

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirement.txt
```

### 2. 准备数据集

将数据按以下结构组织:

```
data/portrait/
├── images/        # 原始RGB图片
└── masks/         # 对应的alpha蒙版 (灰度图,值范围0-1)
```

### 3. 训练模型

```bash
# 基础训练 (30轮)
python train.py --epochs 30

# 优化训练 (50轮,更好的边缘效果)
python train.py --epochs 50

# 自定义参数
python train.py --epochs 50 --batch 16 --lr 1e-4 --size 320
```

**训练参数说明:**
- `--data_root`: 数据集路径 (默认: `./data/portrait/`)
- `--epochs`: 训练轮数 (默认: 50)
- `--batch`: 批次大小 (默认: 8)
- `--lr`: 学习率 (默认: 1e-4)
- `--size`: 输入图片尺寸 (默认: 320)
- `--ckpt_dir`: checkpoint保存路径 (默认: `./checkpoints/`)

### 4. 推理 - 抠图

```bash
# 基本抠图
python infer.py --model ./checkpoints/best.pth --img your_image.jpg --out ./output/

# 批量处理
python infer.py --model ./checkpoints/best.pth --img folder/*.jpg --out ./output/
```

**输出文件:**
- `mask.png` - 灰度mask (前景白色,背景黑色)
- `cutout.png` - 透明背景抠图 (RGBA格式,可直接使用)
- `composite.png` - 黑色背景合成预览

### 5. 背景替换

```bash
# 替换背景
python replace_bg.py --model ./checkpoints/best.pth \
                     --img your_photo.jpg \
                     --background new_background.jpg \
                     --out ./replaced_output/

# 添加边缘平滑
python replace_bg.py --model ./checkpoints/best.pth \
                     --img your_photo.jpg \
                     --background new_background.jpg \
                     --smooth 1.5 \
                     --out ./replaced_output/
```

**参数说明:**
- `--img`: 需要替换背景的图片
- `--background`: 新背景图片
- `--smooth`: 边缘平滑系数 (0=不平滑, 1-3=轻微平滑)

## 📂 项目结构

```
.
├── model.py              # UNet模型定义 (ResNet18编码器 + Attention)
├── dataset.py            # 数据集加载和增强
├── losses.py             # 损失函数 (BCE + L1 + Edge + Gradient)
├── train.py              # 训练脚本
├── infer.py              # 推理脚本 (抠图)
├── replace_bg.py         # 背景替换脚本
├── utils.py              # 工具函数 (保存checkpoint等)
├── requirement.txt       # 依赖包列表
├── README.md             # 项目说明
└── checkpoints/          # 模型checkpoint保存目录
```

## 🔧 模型架构

### 核心组件

1. **编码器**: ResNet18 (预训练) - 提取多尺度特征
2. **解码器**: 上采样 + 跳跃连接 - 恢复空间分辨率
3. **注意力机制**: Attention Gates - 聚焦前景区域
4. **激活函数**: Sigmoid - 输出0-1的alpha值

### 损失函数

组合损失 = BCE + 10×L1 + 15×Edge + 20×Gradient

- **BCE Loss**: 二分类基础损失
- **L1 Loss**: 整体精度优化
- **Edge Loss**: Laplacian边缘损失
- **Gradient Loss**: Sobel梯度损失 (新增,专注边缘细节)

### 数据增强

- 水平翻转 (p=0.5)
- 亮度对比度调整 (p=0.5)
- 色彩抖动 (p=0.5)
- 高斯噪声 (p=0.3)
- 随机模糊 (p=0.2)
- Gamma变换 (p=0.3)
- HSV调整 (p=0.3)

### 训练策略

- **优化器**: Adam (lr=1e-4)
- **学习率调度**: MultiStepLR (在第20和40轮时lr×0.1)
- **批次大小**: 8 (可根据显存调整)
- **输入尺寸**: 320×320

## 📊 性能指标

在验证集上的表现:
- **MAE**: ~0.014 (Mean Absolute Error)
- **训练轮数**: 50 epochs
- **训练速度**: ~10 it/s (单个RTX GPU)

## 🎨 使用示例

### 示例1: 证件照背景替换

```bash
# 将证件照背景替换为纯色
python replace_bg.py --model ./checkpoints/best.pth \
                     --img id_photo.jpg \
                     --background blue_bg.jpg \
                     --out ./id_output/
```

### 示例2: 产品图抠图

```bash
# 生成透明背景产品图
python infer.py --model ./checkpoints/best.pth \
                --img product.jpg \
                --out ./product_output/
```

## 🛠️ 高级用法

### 从checkpoint恢复训练

修改 `train.py` 添加resume功能:

```python
# 加载之前的checkpoint继续训练
checkpoint = torch.load('./checkpoints/best.pth')
model.load_state_dict(checkpoint['model'])
optimizer.load_state_dict(checkpoint['opt'])
start_epoch = checkpoint['epoch'] + 1
```

### 调整损失函数权重

在 `losses.py` 中修改权重:

```python
def composite_loss(pred, target):
    l_bce = bce(pred, target)
    l_l1 = F.l1_loss(pred, target)
    l_edge = laplacian_edge(pred - target)
    l_grad = gradient_loss(pred, target)
    
    # 自定义权重
    return l_bce + 10 * l_l1 + 15 * l_edge + 20 * l_grad
```

## 📝 注意事项

1. **NumPy版本**: 需要 NumPy < 2.0 (兼容性问题)
   ```bash
   pip install "numpy<2"
   ```

2. **显存要求**: 
   - 训练: 至少4GB显存 (batch_size=8)
   - 推理: 约2GB显存

3. **数据格式**:
   - 输入图片: RGB, 任意尺寸
   - Mask: 灰度图, 值范围[0, 1], 与图片同尺寸

## 🤝 贡献

欢迎提交 Issue 和 Pull Request!

## 📄 License

MIT License

## 🙏 致谢

- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- UNet: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- Attention Gates: [Attention U-Net](https://arxiv.org/abs/1804.03999)

## 📧 联系方式

如有问题,请提交 [GitHub Issue](https://github.com/your-username/portrait-matting/issues)
