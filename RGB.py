

import os
import cv2
import numpy as np

# =========================
# 路径配置
# =========================
original_dir = r"/home/xxy/Desktop/traindata3/train3/"
screen_dir = r"/home/xxy/Desktop/traindata3/noise/"
residual_dir = r"/home/xxy/Desktop/traindata3/RGB_residual/"

os.makedirs(residual_dir, exist_ok=True)

TARGET_SIZE = (256, 256)

image_extensions = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}

# =========================
# 获取文件
# =========================
original_files = {
    os.path.splitext(f)[0]: f
    for f in os.listdir(original_dir)
    if os.path.splitext(f)[1].lower() in image_extensions
}

screen_files = {
    os.path.splitext(f)[0]: f
    for f in os.listdir(screen_dir)
    if os.path.splitext(f)[1].lower() in image_extensions
}

# =========================
# 同名匹配
# =========================
common_names = sorted(
    set(original_files.keys()) &
    set(screen_files.keys())
)

print(f"原始图像数量: {len(original_files)}")
print(f"屏摄图像数量: {len(screen_files)}")
print(f"匹配图像数量: {len(common_names)}")

# =========================
# 处理
# =========================
success_count = 0

for name in common_names:

    original_path = os.path.join(
        original_dir,
        original_files[name]
    )

    screen_path = os.path.join(
        screen_dir,
        screen_files[name]
    )

    # 读取
    original = cv2.imread(original_path)
    screen = cv2.imread(screen_path)

    if original is None or screen is None:
        print(f"[跳过] 无法读取: {name}")
        continue

    # =========================
    # 统一 resize 到 256×256
    # =========================
    original = cv2.resize(
        original,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA
    )

    screen = cv2.resize(
        screen,
        TARGET_SIZE,
        interpolation=cv2.INTER_AREA
    )

    # =========================
    # 转 float32
    # =========================
    original = original.astype(np.float32)
    screen = screen.astype(np.float32)

    # =========================
    # Screen - Original
    # =========================
    residual = screen - original

    # =========================
    # [-255,255] -> [0,255]
    # =========================
    residual_jpg = (residual + 255.0) / 2.0

    residual_jpg = np.clip(
        residual_jpg,
        0,
        255
    ).astype(np.uint8)

    # =========================
    # 保存 JPG
    # =========================
    save_path = os.path.join(
        residual_dir,
        name + ".jpg"
    )

    cv2.imwrite(
        save_path,
        residual_jpg,
        [cv2.IMWRITE_JPEG_QUALITY, 100]
    )

    success_count += 1

    print(
        f"[{success_count}/{len(common_names)}] "
        f"{name} | "
        f"Residual range: "
        f"{residual.min():.2f} ~ {residual.max():.2f}"
    )

print("\n==============================")
print("处理完成")
print(f"成功生成: {success_count} 张")
print(f"尺寸: 256 × 256")
print(f"保存目录: {residual_dir}")
print("==============================")