# infer.py - Portrait matting inference script
# Function: Segment portrait from input image, generate transparent background cutout  
# Output: mask.png (grayscale mask), cutout.png (transparent background), composite.png (black background preview)

import torch
from model import UNet
from PIL import Image
import numpy as np
import os
import torch.nn.functional as F

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available, will skip comparison.png")

def run_infer(model_path, img_path, out_path, size=320, device='cuda', keep_ratio=False):
    """
    Run portrait matting inference on a single image
    
    Args:
        model_path: Path to model checkpoint, e.g. './checkpoints/best.pth'
        img_path: Path to input image
        out_path: Path to output folder
        size: Image size for inference (model input)
        device: 'cuda' or 'cpu'
        keep_ratio: Whether to keep original aspect ratio (True=padding, False=stretch)
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet(n_classes=1, use_attention=True, pretrained=False).to(device)
    ck = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(ck['model'])
    model.eval()
    print(f"Loaded model from {model_path} (epoch {ck.get('epoch', 'N/A')})")
    
    # Read original image
    orig_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = orig_img.size
    print(f"Input image: {img_path} ({orig_w}x{orig_h})")
    
    # Prepare model input (use exactly same preprocessing as training)
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    
    # Consistent preprocessing with dataset.py
    transform = A.Compose([
        A.Resize(size, size),
        ToTensorV2(),
    ])
    
    # Important: albumentations requires numpy uint8 array as input
    img_resized = orig_img.resize((size, size), Image.BILINEAR)
    img_arr = np.array(img_resized)  # HWC, uint8 [0-255]
    
    # Albumentations transform
    transformed = transform(image=img_arr)
    inp = transformed['image'].unsqueeze(0).float().to(device)  # [1, 3, H, W], float32 [0-255]
    
    print(f"Input to model - min: {inp.min().item():.2f}, max: {inp.max().item():.2f}")
    # ToTensorV2 does not normalize, keeps [0-255] range, consistent with training

    
    # Inference
    with torch.no_grad():
        pred = model(inp)  # [1, 1, H, W]
        print(f"Model output - min: {pred.min().item():.6f}, max: {pred.max().item():.6f}, mean: {pred.mean().item():.6f}")
        # Interpolate back to original image size
        pred_orig_size = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        pred_mask = pred_orig_size[0, 0].cpu().numpy()  # [orig_h, orig_w]
        print(f"Pred mask - min: {pred_mask.min():.6f}, max: {pred_mask.max():.6f}, mean: {pred_mask.mean():.6f}")
    
    # Convert back to original image size numpy array
    orig_img_arr = np.array(orig_img).astype('float32') / 255.0
    
    # Save results
    os.makedirs(out_path, exist_ok=True)
    
    # 1. Save predicted mask (grayscale)
    mask_uint8 = (pred_mask * 255).astype('uint8')
    Image.fromarray(mask_uint8).save(os.path.join(out_path, 'mask.png'))
    
    # 2. Save cutout result (foreground + transparent background, RGBA)
    rgba = np.concatenate([
        (orig_img_arr * 255).astype('uint8'),
        (pred_mask[..., None] * 255).astype('uint8')
    ], axis=2)
    Image.fromarray(rgba, mode='RGBA').save(os.path.join(out_path, 'cutout.png'))
    
    # 3. Save composite (foreground + black background, RGB)
    comp = (orig_img_arr * pred_mask[..., None] * 255).astype('uint8')
    Image.fromarray(comp).save(os.path.join(out_path, 'composite.png'))
    
    # 4. Save comparison image (original|mask|cutout)
    if HAS_MPL:
        try:
            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(orig_img); axes[0].set_title('Original'); axes[0].axis('off')
            axes[1].imshow(pred_mask, cmap='gray'); axes[1].set_title('Predicted Mask'); axes[1].axis('off')
            axes[2].imshow(rgba); axes[2].set_title('Cutout (RGBA)'); axes[2].axis('off')
            axes[3].imshow(comp); axes[3].set_title('Composite'); axes[3].axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, 'comparison.png'), dpi=150)
            plt.close()
            print(f"  - comparison.png (original|mask|cutout comparison)")
        except Exception as e:
            print(f"  (skipped comparison.png: {e})")
    
    print(f"\n✓ Saved results to: {out_path}/")
    print(f"  - mask.png (grayscale mask)")
    print(f"  - cutout.png (transparent background cutout, ready to use)")
    print(f"  - composite.png (black background composite)")
    
    return pred_mask

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--img", required=True)
    parser.add_argument("--out", default="./out/")
    parser.add_argument("--size", type=int, default=320)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    run_infer(args.model, args.img, args.out, size=args.size)
