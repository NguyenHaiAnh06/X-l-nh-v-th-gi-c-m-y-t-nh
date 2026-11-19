import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from chuan_hoa_loc import chuan_hoa_va_loc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def tach_nen_kmeans(img_clean, k=2):
    pixel_values = img_clean.reshape((-1, 1)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv2.kmeans(
        pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    labels_2d = labels.reshape(img_clean.shape)
    goc = [labels_2d[0, 0], labels_2d[0, -1], labels_2d[-1, 0], labels_2d[-1, -1]]
    nhan_nen = max(set(goc), key=goc.count)

    _, cc_labels = cv2.connectedComponents(np.uint8(labels_2d == nhan_nen))
    goc_cc = {cc_labels[0, 0], cc_labels[0, -1], cc_labels[-1, 0], cc_labels[-1, -1]} - {0}
    bg_mask = np.isin(cc_labels, list(goc_cc)).astype(np.uint8) * 255

    mask = cv2.bitwise_not(bg_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    img_tach_nen = cv2.bitwise_and(img_clean, img_clean, mask=mask)
    return img_tach_nen, mask

if __name__ == "__main__":
    mau_clean = chuan_hoa_va_loc(os.path.join(BASE_DIR, "images", "anh_mau.jpg"))
    loi_clean = chuan_hoa_va_loc(os.path.join(BASE_DIR, "images", "anh_loi.jpg"))

    mau_nen, mask_mau = tach_nen_kmeans(mau_clean)
    loi_nen, mask_loi = tach_nen_kmeans(loi_clean)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8))
    axes[0, 0].imshow(mask_mau, cmap="gray")
    axes[0, 0].set_title("Mat na K-means (Mau)")
    axes[0, 1].imshow(mau_nen, cmap="gray")
    axes[0, 1].set_title("Buoc 2: Tach nen (Mau)")

    axes[1, 0].imshow(mask_loi, cmap="gray")
    axes[1, 0].set_title("Mat na K-means (Kiem tra)")
    axes[1, 1].imshow(loi_nen, cmap="gray")
    axes[1, 1].set_title("Buoc 2: Tach nen (Kiem tra)")

    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()