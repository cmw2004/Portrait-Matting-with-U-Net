# train.py
import os
import argparse
from torch.utils.data import DataLoader
import torch
from model import UNet
from dataset import PortraitMattingDataset
from losses import composite_loss
from utils import save_checkpoint, visualize_sample
from tqdm import tqdm
import torchvision

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(n_classes=1, use_attention=True, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Add learning rate scheduler: reduce LR at epoch 20 and 40
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    
    # treat empty string as None to disable background augmentation
    train_bg = args.bg_dir if getattr(args, 'bg_dir', None) else None
    train_ds = PortraitMattingDataset(args.data_root, size=(args.size,args.size), bg_dir=train_bg)
    val_ds = PortraitMattingDataset(args.data_root, size=(args.size,args.size), bg_dir=None)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2)
    start_epoch = 0
    best_mae = 1e9

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for imgs, masks, names in pbar:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            # ensure prediction and mask spatial sizes match (model may output lower-res)
            if preds.shape[2:] != masks.shape[2:]:
                preds = torch.nn.functional.interpolate(preds, size=masks.shape[2:], mode='bilinear', align_corners=False)
            loss = composite_loss(preds, masks)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / (pbar.n + 1), 'lr': optimizer.param_groups[0]['lr']})
        
        # Update learning rate
        scheduler.step()
        
        # validation
        model.eval()
        total_mae = 0.0
        with torch.no_grad():
            for imgs, masks, names in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                preds = model(imgs)
                if preds.shape[2:] != masks.shape[2:]:
                    preds = torch.nn.functional.interpolate(preds, size=masks.shape[2:], mode='bilinear', align_corners=False)
                mae = torch.mean(torch.abs(preds - masks)).item()
                total_mae += mae
        avg_mae = total_mae / len(val_loader)
        print(f"Epoch {epoch} val MAE: {avg_mae:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        # checkpoint
        ck_path = os.path.join(args.ckpt_dir, f"model_epoch{epoch}.pth")
        save_checkpoint(model, optimizer, epoch, ck_path)
        if avg_mae < best_mae:
            best_mae = avg_mae
            save_checkpoint(model, optimizer, epoch, os.path.join(args.ckpt_dir, "best.pth"))
        # save sample visualization
        sample_img, sample_mask, _ = next(iter(val_loader))
        sample_pred = model(sample_img.to(device))
        if sample_pred.shape[2:] != sample_mask.shape[2:]:
            sample_pred = torch.nn.functional.interpolate(sample_pred, size=sample_mask.shape[2:], mode='bilinear', align_corners=False)
        visualize_sample(sample_img[0], sample_mask[0], sample_pred[0].cpu(), save_path=os.path.join(args.ckpt_dir, f"sample_epoch{epoch}.png"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/portrait/")
    parser.add_argument("--bg_dir", type=str, default=None, help='Background images folder path; pass empty string or omit to disable background augmentation')
    parser.add_argument("--ckpt_dir", type=str, default="./checkpoints/")
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)  # Increased to 50 epochs
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    os.makedirs(args.ckpt_dir, exist_ok=True)
    train(args)
