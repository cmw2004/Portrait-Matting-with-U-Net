# replace_bg.py - Background replacement script
# Function: Segment portrait from original image and composite onto new background
# Supports edge smoothing for more natural results

import os
import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from model import UNet
import albumentations as A
from albumentations.pytorch import ToTensorV2

def replace_background(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = UNet(n_classes=1, use_attention=True, pretrained=False).to(device)
    ckpt = torch.load(args.model, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"Loaded model from {args.model} (epoch {ckpt.get('epoch', '?')})")
    
    # Load original image
    orig_img = Image.open(args.img).convert('RGB')
    orig_w, orig_h = orig_img.size
    print(f"Input image: {args.img} ({orig_w}x{orig_h})")
    
    # Load background image
    bg_img = Image.open(args.background).convert('RGB')
    bg_img = bg_img.resize((orig_w, orig_h), Image.BILINEAR)
    print(f"Background: {args.background} (resized to {orig_w}x{orig_h})")
    
    # Prepare input
    size = args.size
    transform = A.Compose([
        A.Resize(size, size),
        ToTensorV2(),
    ])
    
    img_resized = orig_img.resize((size, size), Image.BILINEAR)
    img_arr = np.array(img_resized)
    
    transformed = transform(image=img_arr)
    inp = transformed['image'].unsqueeze(0).float().to(device)
    
    # Inference
    with torch.no_grad():
        pred = model(inp)
        # Interpolate back to original size
        pred_orig_size = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=False)
        pred_mask = pred_orig_size[0, 0].cpu().numpy()
    
    # Convert to numpy arrays
    orig_img_arr = np.array(orig_img).astype('float32')
    bg_img_arr = np.array(bg_img).astype('float32')
    
    # Alpha blending: result = foreground * alpha + background * (1 - alpha)
    alpha = pred_mask[..., None]  # [H, W, 1]
    
    # Optional: smooth edges
    if args.smooth > 0:
        from scipy.ndimage import gaussian_filter
        alpha = gaussian_filter(alpha, sigma=args.smooth)
        alpha = np.clip(alpha, 0, 1)
    
    result = orig_img_arr * alpha + bg_img_arr * (1 - alpha)
    result = result.astype('uint8')
    
    # Save results
    os.makedirs(args.out, exist_ok=True)
    
    # 1. Save image with replaced background
    result_img = Image.fromarray(result)
    result_path = os.path.join(args.out, 'replaced_bg.png')
    result_img.save(result_path)
    
    # 2. Save mask
    mask_uint8 = (pred_mask * 255).astype('uint8')
    mask_path = os.path.join(args.out, 'mask.png')
    Image.fromarray(mask_uint8).save(mask_path)
    
    # 3. Save comparison image (original|mask|new background)
    cmp_h = 512
    cmp_w = int(512 * orig_w / orig_h)
    mask_rgb = np.stack([mask_uint8]*3, axis=2)
    comparison = np.hstack([
        np.array(orig_img.resize((cmp_w, cmp_h))),
        np.array(Image.fromarray(mask_rgb).resize((cmp_w, cmp_h))),
        np.array(result_img.resize((cmp_w, cmp_h)))
    ])
    comparison_path = os.path.join(args.out, 'comparison.png')
    Image.fromarray(comparison).save(comparison_path)
    
    print(f"\n✓ Saved results to: {args.out}/")
    print(f"  - replaced_bg.png (image with replaced background)")
    print(f"  - mask.png (portrait mask)")
    print(f"  - comparison.png (original|mask|new background comparison)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Replace image background')
    parser.add_argument("--model", type=str, required=True, help='Path to model checkpoint')
    parser.add_argument("--img", type=str, required=True, help='Path to input image')
    parser.add_argument("--background", type=str, required=True, help='Path to background image')
    parser.add_argument("--out", type=str, default="./output_bg/", help='Output folder')
    parser.add_argument("--size", type=int, default=320, help='Model input size')
    parser.add_argument("--smooth", type=float, default=0, help='Edge smoothing factor (0=no smoothing, 1-3=light smoothing)')
    args = parser.parse_args()
    replace_background(args)
