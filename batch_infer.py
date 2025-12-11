# batch_infer.py - 批量推理脚本
# 从验证集中选择多张图片进行推理，生成更多示例

import os
import random
from infer import run_infer

def batch_inference(model_path, image_dir, output_base_dir, num_samples=10, size=320, device='cuda'):
    """
    从验证集中随机选择图片进行批量推理
    
    Args:
        model_path: 模型检查点路径
        image_dir: 验证集图片目录
        output_base_dir: 输出根目录
        num_samples: 推理的图片数量
        size: 模型输入尺寸
        device: 'cuda' 或 'cpu'
    """
    # 获取所有图片
    all_images = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    
    # 随机选择图片（或者按顺序选择前N张）
    if num_samples > len(all_images):
        num_samples = len(all_images)
    
    selected_images = random.sample(all_images, num_samples)
    # 或者选择前N张: selected_images = all_images[:num_samples]
    
    print(f"将处理 {num_samples} 张图片...")
    print(f"从 {len(all_images)} 张验证集图片中选择")
    print("=" * 60)
    
    # 为每张图片创建输出目录并进行推理
    for idx, img_name in enumerate(selected_images, 1):
        img_path = os.path.join(image_dir, img_name)
        
        # 创建以图片名命名的输出目录
        img_basename = os.path.splitext(img_name)[0]
        out_dir = os.path.join(output_base_dir, img_basename)
        
        print(f"\n[{idx}/{num_samples}] 处理: {img_name}")
        print("-" * 60)
        
        try:
            run_infer(model_path, img_path, out_dir, size=size, device=device)
        except Exception as e:
            print(f"❌ 处理失败: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"✓ 批量推理完成！共处理 {num_samples} 张图片")
    print(f"✓ 结果保存在: {output_base_dir}")
    
    # 列出所有生成的目录
    print("\n生成的示例目录:")
    for img_name in selected_images:
        img_basename = os.path.splitext(img_name)[0]
        out_dir = os.path.join(output_base_dir, img_basename)
        if os.path.exists(out_dir):
            print(f"  - {img_basename}/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='批量推理脚本')
    parser.add_argument("--model", default="./checkpoints/best.pth", help="模型路径")
    parser.add_argument("--image_dir", default="./data/P3M-10k/validation/P3M-500-NP/original_image", 
                        help="验证集图片目录")
    parser.add_argument("--output_dir", default="./examples_batch", help="输出根目录")
    parser.add_argument("--num", type=int, default=10, help="生成示例数量")
    parser.add_argument("--size", type=int, default=320, help="模型输入尺寸")
    parser.add_argument("--device", default="cuda", choices=['cuda', 'cpu'], help="运行设备")
    
    args = parser.parse_args()
    
    # 确保路径存在
    if not os.path.exists(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        exit(1)
    
    if not os.path.exists(args.image_dir):
        print(f"❌ 图片目录不存在: {args.image_dir}")
        exit(1)
    
    batch_inference(
        model_path=args.model,
        image_dir=args.image_dir,
        output_base_dir=args.output_dir,
        num_samples=args.num,
        size=args.size,
        device=args.device
    )
