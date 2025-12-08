# utils.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

def save_checkpoint(model, optimizer, epoch, path):
    torch.save({'model': model.state_dict(), 'opt': optimizer.state_dict(), 'epoch': epoch}, path)

def load_checkpoint(model, path, device='cpu'):
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck['model'])
    return ck

def visualize_sample(image_tensor, mask_tensor, pred_tensor, save_path):
    # image_tensor: C,H,W
    img = image_tensor.permute(1,2,0).cpu().numpy()
    gt = mask_tensor.squeeze().cpu().numpy()
    pred = pred_tensor.detach().squeeze().cpu().numpy()
    fig, axes = plt.subplots(1,4,figsize=(12,4))
    axes[0].imshow((img*255).astype('uint8'))
    axes[0].set_title('image'); axes[0].axis('off')
    axes[1].imshow(gt, cmap='gray'); axes[1].set_title('gt'); axes[1].axis('off')
    axes[2].imshow(pred, cmap='gray'); axes[2].set_title('pred'); axes[2].axis('off')
    # composite with black bg
    comp = img*(pred[...,None])
    axes[3].imshow((comp*255).astype('uint8')); axes[3].set_title('composite'); axes[3].axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
