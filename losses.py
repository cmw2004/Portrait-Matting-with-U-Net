# losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

bce = nn.BCELoss()

def laplacian_edge(x):
    # x: [B,1,H,W]
    kernel = torch.tensor([[0,1,0],[1,-4,1],[0,1,0]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
    out = F.conv2d(x, kernel, padding=1)
    return torch.abs(out).mean()

def gradient_loss(pred, target):
    """Calculate gradient loss, specifically optimize edge details"""
    # Sobel operator to compute gradients in x and y directions
    sobel_x = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)
    
    grad_loss_x = F.l1_loss(pred_grad_x, target_grad_x)
    grad_loss_y = F.l1_loss(pred_grad_y, target_grad_y)
    
    return grad_loss_x + grad_loss_y

def composite_loss(pred, target):
    # BCE + L1 + edge + gradient (enhanced edges)
    l_bce = bce(pred, target)
    l_l1 = F.l1_loss(pred, target)
    l_edge = laplacian_edge(pred - target)
    l_grad = gradient_loss(pred, target)
    
    # Adjusted weights: increase edge and gradient loss weights
    return l_bce + 10 * l_l1 + 15 * l_edge + 20 * l_grad
