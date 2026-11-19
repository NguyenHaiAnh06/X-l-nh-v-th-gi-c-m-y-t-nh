import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from chuan_hoa_loc import chuan_hoa_va_loc
from tach_nen import tach_nen_kmeans

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def cat_bien_sobel(img_tach_nen, mask):
    gx = cv2.Sobel(img_tach_nen, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_tach_nen, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(gx**2 + gy**2)
    sobel = np.uint8(np.clip(magnitude, 0, 255))

    mask_ben_trong = cv2.erode(mask, np.ones((5, 5), np.uint8), iterations=2)
    sobel_san_pham = cv2.bitwise_and(sobel, sobel, mask=mask_ben_trong)
    return sobel_san_pham, mask_ben_trong

def phan_tich_va_ve_histogram(sobel_mau, mask_mau, sobel_kt, mask_kt, nguong_t=0.0025):
    _, bien_khung_chuan = cv2.threshold(sobel_mau, 35, 255, cv2.THRESH_BINARY)
    bien_khung_gian = cv2.dilate(bien_khung_chuan, np.ones((7, 7), np.uint8), iterations=1)

    vung_kiem_tra = cv2.bitwise_and(mask_mau, cv2.bitwise_not(bien_khung_gian))
    vung_kiem_tra = cv2.bitwise_and(vung_kiem_tra, mask_kt)

    loi_mau = cv2.bitwise_and(sobel_mau, sobel_mau, mask=vung_kiem_tra)
    loi_kt = cv2.bitwise_and(sobel_kt, sobel_kt, mask=vung_kiem_tra)

    h1 = cv2.calcHist([loi_mau], [0], vung_kiem_tra, [256], [0, 256]).flatten()
    h2 = cv2.calcHist([loi_kt], [0], vung_kiem_tra, [256], [0, 256]).flatten()

    h1_norm = h1 / (np.sum(h1) + 1e-10)
    h2_norm = h2 / (np.sum(h2) + 1e-10)

    chi_square = float(np.sum(((h1_norm - h2_norm) ** 2) / (h1_norm + h2_norm + 1e-10)))
    pixel_xuoc_mau = int(np.sum(h1[35:]))
    pixel_xuoc_kt = int(np.sum(h2[35:]))

    ket_luan = "SAN PHAM LOI" if chi_square > nguong_t else "SAN PHAM DAT CHUAN"
    mau_ket_luan = "#BE123C" if chi_square > nguong_t else "#0F766E"

    print(f"So pixel bat thuong (>35) - Mau: {pixel_xuoc_mau} | Kiem tra: {pixel_xuoc_kt}")
    print(f"Do sai lech Chi-Square (X^2): {chi_square:.6f} (Nguong T = {nguong_t})")
    print(f"Ket luan: {ket_luan}")

    fig = plt.figure(figsize=(12, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(sobel_mau, cmap="gray")
    ax1.set_title("1. Bien Sobel - Anh mau")
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.imshow(sobel_kt, cmap="gray")
    ax2.set_title("2. Bien Sobel - Anh kiem tra")
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[0, 1])
    ax3.imshow(loi_kt, cmap="hot")
    ax3.set_title("3. Tach rieng vet loi be mat")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 1])
    nhom = ["Xuoc nhe\n(35-99)", "Xuoc trung binh\n(100-179)", "Xuoc sau\n(180-255)"]
    val_mau = [np.sum(h1_norm[35:100]) * 100, np.sum(h1_norm[100:180]) * 100, np.sum(h1_norm[180:]) * 100]
    val_kt = [np.sum(h2_norm[35:100]) * 100, np.sum(h2_norm[100:180]) * 100, np.sum(h2_norm[180:]) * 100]
    x = np.arange(len(nhom))
    ax4.bar(x - 0.18, val_mau, width=0.35, color="#0F766E", label="Mau (H1)")
    ax4.bar(x + 0.18, val_kt, width=0.35, color="#BE123C", label="Kiem tra (H2)")
    ax4.set_xticks(x)
    ax4.set_xticklabels(nhom, fontsize=8)
    ax4.set_ylabel("Ty le dien tich loi (%)")
    ax4.set_title("4. Phan bo muc do vet xuoc")
    ax4.legend(fontsize=8)

    ax_hist = fig.add_subplot(gs[:, 2])
    bins = np.arange(35, 256)
    ax_hist.plot(bins, h1_norm[35:], color="#0F766E", linewidth=2, label="Histogram Mau (H1)")
    ax_hist.plot(bins, h2_norm[35:], color="#BE123C", linewidth=2, label="Histogram Kiem tra (H2)")
    ax_hist.fill_between(bins, h1_norm[35:], h2_norm[35:], color="#FDA4AF", alpha=0.45)
    ax_hist.set_title(
        f"5. SO SANH HISTOGRAM BIEN LOI (35 - 255)\nChi-Square = {chi_square:.5f} | Nguong T = {nguong_t}\n=> {ket_luan}",
        fontsize=10,
        fontweight="bold",
        color=mau_ket_luan,
    )
    ax_hist.set_xlabel("Cuong do bien bat thuong (35 - 255)")
    ax_hist.set_ylabel("Tan suat chuan hoa")
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    mau_clean = chuan_hoa_va_loc(os.path.join(BASE_DIR, "images", "anh_mau.jpg"))
    loi_clean = chuan_hoa_va_loc(os.path.join(BASE_DIR, "images", "anh_loi.jpg"))

    mau_nen, mask_mau = tach_nen_kmeans(mau_clean)
    loi_nen, mask_loi = tach_nen_kmeans(loi_clean)

    sobel_mau, mask_bien_mau = cat_bien_sobel(mau_nen, mask_mau)
    sobel_loi, mask_bien_loi = cat_bien_sobel(loi_nen, mask_loi)

    phan_tich_va_ve_histogram(sobel_mau, mask_bien_mau, sobel_loi, mask_bien_loi, nguong_t=0.0025)