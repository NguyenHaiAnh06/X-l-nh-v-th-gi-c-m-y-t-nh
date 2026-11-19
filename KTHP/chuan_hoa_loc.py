import os
import cv2
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def chuan_hoa_va_loc(duong_dan, kich_thuoc=(400, 400)):
    img = cv2.imread(duong_dan, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Khong tim thay anh: {duong_dan}")

    img_resized = cv2.resize(img, kich_thuoc, interpolation=cv2.INTER_AREA)
    img_norm = cv2.normalize(img_resized, None, 0, 255, cv2.NORM_MINMAX)
    img_median = cv2.medianBlur(img_norm, 3)
    img_clean = cv2.GaussianBlur(img_median, (3, 3), 0)
    return img_clean

if __name__ == "__main__":
    path_mau = os.path.join(BASE_DIR, "images", "anh_mau.jpg")
    path_loi = os.path.join(BASE_DIR, "images", "anh_loi.jpg")

    mau_clean = chuan_hoa_va_loc(path_mau)
    loi_clean = chuan_hoa_va_loc(path_loi)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].imshow(mau_clean, cmap="gray")
    axes[0].set_title("Buoc 1: Anh mau sau chuan hoa & loc")
    axes[0].axis("off")

    axes[1].imshow(loi_clean, cmap="gray")
    axes[1].set_title("Buoc 1: Anh kiem tra sau chuan hoa & loc")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()