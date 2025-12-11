# 批量推理结果说明

## 生成的示例

本次从P3M-10k验证集中随机选择了10张图片进行推理，生成了以下示例：

### 生成的图片列表
1. **p_930f61e7** - 1588x1080
2. **p_d2cbcaaf** - 1440x1080
3. **p_87564383** - 1080x1224
4. **p_b7a2fc3c** - 1920x1080
5. **p_c770e8ba** - 1727x1080
6. **p_fb23fc85** - 1080x1617
7. **p_48f2de97** - 1728x1080
8. **p_e7028c5f** - 1617x1080
9. **p_cca19cb7** - 1080x1620
10. **p_127746a6** - 1920x1080

## 每个示例包含的文件

每个示例目录中包含4个文件：
- `mask.png` - 灰度掩码图（Alpha matte）
- `cutout.png` - RGBA透明背景抠图
- `composite.png` - 黑色背景合成图
- `comparison.png` - 对比图（原图|掩码|抠图|合成）

## 如何生成更多示例

使用批量推理脚本：

```bash
python batch_infer.py --model ./checkpoints/best.pth --num 10
```

参数说明：
- `--model`: 模型检查点路径（默认：./checkpoints/best.pth）
- `--image_dir`: 验证集图片目录（默认：./data/P3M-10k/validation/P3M-500-NP/original_image）
- `--output_dir`: 输出根目录（默认：./examples_batch）
- `--num`: 生成示例数量（默认：10）
- `--size`: 模型输入尺寸（默认：320）
- `--device`: 运行设备，cuda或cpu（默认：cuda）

## 网站展示

这些对比图已经被添加到项目网站的Results部分，每张图片展示了：
- 原始输入图像
- 预测的Alpha掩码
- 透明背景抠图结果
- 黑色背景合成预览

访问 `index.html` 查看完整展示。
