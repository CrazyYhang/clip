"""
Flickr30k 数据集下载脚本

使用 kagglehub 下载 Flickr30k 数据集（优先），
或通过 HuggingFace Datasets 作为备选方案。

用法:
    python download_flickr30k.py

下载后目录结构:
    data/flickr30k/
    ├── images/           # 31783 张 JPEG 图片
    └── captions.csv      # 每张图片 5 条英文描述
"""

import os
import csv
import shutil
import glob
import argparse
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data" / "flickr30k"
IMAGES_DIR = DATA_DIR / "images"


def download_via_kagglehub():
    """方法 1: 使用 kagglehub 从 Kaggle 下载 Flickr30k"""
    print("[方法 1] 尝试通过 kagglehub 下载...")

    try:
        import kagglehub
    except ImportError:
        print("  kagglehub 未安装, 正在安装...")
        import subprocess
        subprocess.check_call(["pip", "install", "kagglehub", "-q"])
        import kagglehub

    # 下载 Flickr30k 数据集 (hsankesara 版本, 包含图片和 CSV)
    print("  正在从 Kaggle 下载 Flickr30k (约 4.8GB, 请耐心等待)...")
    path = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
    print(f"  下载完成, 路径: {path}")

    # 检查下载内容
    downloaded = Path(path)
    print(f"  下载内容: {list(downloaded.iterdir())}")

    return organize_kaggle_download(downloaded)


def download_via_huggingface():
    """方法 2: 使用 HuggingFace Datasets 下载 Flickr30k"""
    print("[方法 2] 尝试通过 HuggingFace Datasets 下载...")

    try:
        from datasets import load_dataset
    except ImportError:
        print("  datasets 未安装, 正在安装...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "-q"])
        from datasets import load_dataset

    dataset = load_dataset("nlphuji/flickr30k", split="test")
    print(f"  数据集加载成功, 包含 {len(dataset)} 条记录")

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    captions_data = []
    for idx, item in enumerate(dataset):
        image_name = f"{idx:010d}.jpg"
        image_path = IMAGES_DIR / image_name

        # 保存图片
        if not image_path.exists():
            item["image"].save(str(image_path))

        # 收集描述
        for cap_idx, caption in enumerate(item["caption"]):
            captions_data.append([image_name, cap_idx, caption])

        if (idx + 1) % 5000 == 0:
            print(f"  已处理 {idx + 1}/{len(dataset)} 条记录...")

    # 保存 captions.csv
    captions_path = DATA_DIR / "captions.csv"
    with open(captions_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "caption_number", "caption"])
        writer.writerows(captions_data)

    print(f"  完成! {len(dataset)} 张图片, {len(captions_data)} 条描述")
    print(f"  图片目录: {IMAGES_DIR}")
    print(f"  描述文件: {captions_path}")
    return True


def organize_kaggle_download(downloaded: Path):
    """整理 kagglehub 下载的文件"""
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    # 查找 CSV 文件 (可能是 flickr30k_images/results.csv)
    csv_files = list(downloaded.glob("**/*.csv"))
    if not csv_files:
        csv_files = list(downloaded.glob("**/results*"))

    print(f"  找到 CSV 文件: {csv_files}")

    # 查找图片目录
    image_dirs = [d for d in downloaded.rglob("*") if d.is_dir() and d != downloaded]
    image_files = list(downloaded.rglob("*.jpg")) + list(downloaded.rglob("*.png"))

    print(f"  找到图片文件: {len(image_files)} 个")

    # 复制/移动图片到 images/ 目录
    for img_file in image_files:
        dest = IMAGES_DIR / img_file.name
        if not dest.exists():
            shutil.copy2(img_file, dest)

    # 整理 CSV 描述文件
    if csv_files:
        csv_path = csv_files[0]
        print(f"  CSV 文件内容前 3 行:")
        with open(csv_path, "r") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"    {line.strip()}")

        dest_csv = DATA_DIR / "captions.csv"
        shutil.copy2(csv_path, dest_csv)
        print(f"  描述文件已复制到: {dest_csv}")
    else:
        print("  注意: 未找到 CSV 描述文件, 请手动整理")

    print(f"  图片已复制到: {IMAGES_DIR}")
    return True


def verify_dataset():
    """验证下载的数据集"""
    images = list(IMAGES_DIR.glob("*.jpg")) + list(IMAGES_DIR.glob("*.png"))
    captions_csv = DATA_DIR / "captions.csv"

    if not images:
        print("[错误] 未找到图片文件! 请检查 data/flickr30k/images/ 目录")
        return False

    if not captions_csv.exists():
        print("[错误] 未找到 captions.csv!")
        return False

    # 读取 captions.csv 统计
    with open(captions_csv, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        rows = list(reader)

    unique_images_in_csv = set(row[0] for row in rows)

    print(f"\n=== 数据集验证结果 ===")
    print(f"  图片文件数: {len(images)}")
    print(f"  描述条目数: {len(rows)}")
    print(f"  CSV 中唯一图片数: {len(unique_images_in_csv)}")
    if rows:
        print(f"  每张图片描述数: 约 {len(rows) / max(len(unique_images_in_csv), 1):.1f}")

    # 检查一致性
    image_names = {img.name for img in images}
    csv_only = unique_images_in_csv - image_names
    images_only = image_names - unique_images_in_csv

    if csv_only:
        print(f"  警告: {len(csv_only)} 张图片在 CSV 中但不在 images/ 中")
        if len(csv_only) <= 5:
            print(f"    缺失图片: {csv_only}")
    if images_only:
        print(f"  警告: {len(images_only)} 张图片在 images/ 中但不在 CSV 中")

    if not csv_only and not images_only:
        print("  图片与描述文件完全匹配 ✓")

    return True


def main():
    parser = argparse.ArgumentParser(description="下载 Flickr30k 数据集")
    parser.add_argument("--method", choices=["kaggle", "hf", "auto"],
                        default="auto", help="下载方式 (默认: auto)")
    parser.add_argument("--verify-only", action="store_true",
                        help="仅验证已下载的数据集")
    args = parser.parse_args()

    if args.verify_only:
        verify_dataset()
        return

    print("=" * 60)
    print("Flickr30k 数据集下载工具")
    print("=" * 60)
    print(f"目标目录: {DATA_DIR}")
    print()

    success = False

    if args.method in ("auto", "kaggle"):
        try:
            success = download_via_kagglehub()
        except Exception as e:
            print(f"  kagglehub 下载失败: {e}")
            if args.method == "kaggle":
                print("[失败] 请检查 Kaggle 账号设置或网络连接")
                return

    if not success and args.method in ("auto", "hf"):
        try:
            success = download_via_huggingface()
        except Exception as e:
            print(f"  HuggingFace 下载失败: {e}")

    if success:
        print("\n" + "=" * 60)
        verify_dataset()
        print("=" * 60)
        print("\n下载完成! 可以运行 python my_clip/dataset.py 测试数据加载")
    else:
        print("\n[失败] 所有下载方式均失败")
        print("请手动从以下地址下载:")
        print("  1. Kaggle: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset")
        print("  2. HuggingFace: https://huggingface.co/datasets/nlphuji/flickr30k")
        print(f"  下载后请将图片放入 {IMAGES_DIR}")
        print(f"  并将描述文件保存为 {DATA_DIR / 'captions.csv'}")


if __name__ == "__main__":
    main()
