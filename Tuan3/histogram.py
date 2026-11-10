import cv2
import matplotlib.pyplot as plt
import numpy as np

anh_mau_goc = cv2.imread("file_anh/Screenshot 2026-10-31 at 22.10.37.png")

chieu_cao, chieu_rong, so_kenh = anh_mau_goc.shape


def tinh_histogram_thu_cong(ma_tran_kenh):
    h, w = ma_tran_kenh.shape
    bieu_do = np.zeros(256, dtype=int)
    for i in range(h):
        for j in range(w):
            gia_tri_pixel = ma_tran_kenh[i, j]
            bieu_do[gia_tri_pixel] += 1
    return bieu_do


# histo 
kenh_xanh_duong = np.zeros((chieu_cao, chieu_rong), dtype=np.uint8)
kenh_xanh_la = np.zeros((chieu_cao, chieu_rong), dtype=np.uint8)
kenh_do = np.zeros((chieu_cao, chieu_rong), dtype=np.uint8)

for i in range(chieu_cao):
    for j in range(chieu_rong):
        kenh_xanh_duong[i, j] = anh_mau_goc[i, j, 0]
        kenh_xanh_la[i, j] = anh_mau_goc[i, j, 1]
        kenh_do[i, j] = anh_mau_goc[i, j, 2]

his_xanh_duong = tinh_histogram_thu_cong(kenh_xanh_duong)
his_xanh_la = tinh_histogram_thu_cong(kenh_xanh_la)
his_do = tinh_histogram_thu_cong(kenh_do)

plt.figure(figsize=(10, 5))
plt.plot(his_do, color="red", label="Kênh Đỏ")
plt.plot(his_xanh_la, color="green", label="Kênh Xanh Lá")
plt.plot(his_xanh_duong, color="blue", label="Kênh Xanh Dương")
plt.title("Biểu đồ Histogram của 3 kênh màu")
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")
plt.legend()
plt.grid(True)
plt.show()


# tbc
anh_xam = np.zeros((chieu_cao, chieu_rong), dtype=np.uint8)
for i in range(chieu_cao):
    for j in range(chieu_rong):
        trung_binh = (
            int(anh_mau_goc[i, j, 0])
            + int(anh_mau_goc[i, j, 1])
            + int(anh_mau_goc[i, j, 2])
        ) // 3
        anh_xam[i, j] = np.uint8(trung_binh)

cv2.imshow("Anh Xam Trung Binh Cong", anh_xam)
cv2.waitKey(0)

# ve
his_anh_xam = tinh_histogram_thu_cong(anh_xam)

plt.figure(figsize=(10, 5))
plt.plot(his_anh_xam, color="black", label="Ảnh xám")
plt.title("Biểu đồ Histogram ảnh xám")
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")
plt.legend()
plt.grid(True)
plt.show()


# can bang histogram
def can_bang_histogram_thu_cong(ma_tran_kenh):
    h, w = ma_tran_kenh.shape
    tong_so_pixel = h * w
    bieu_do = tinh_histogram_thu_cong(ma_tran_kenh)

    cdf = np.zeros(256, dtype=float)
    tong_tich_luy = 0
    for k in range(256):
        tong_tich_luy += bieu_do[k]
        cdf[k] = tong_tich_luy

    cdf_nho_nhat = 0
    for k in range(256):
        if cdf[k] > 0:
            cdf_nho_nhat = cdf[k]
            break

    bang_anh_xa = np.zeros(256, dtype=np.uint8)
    for k in range(256):
        if tong_so_pixel - cdf_nho_nhat > 0:
            gia_tri_moi = round(
                ((cdf[k] - cdf_nho_nhat) / (tong_so_pixel - cdf_nho_nhat)) * 255
            )
        else:
            gia_tri_moi = round((cdf[k] / tong_so_pixel) * 255)
        bang_anh_xa[k] = np.clip(gia_tri_moi, 0, 255)

    anh_da_can_bang = np.zeros((h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            anh_da_can_bang[i, j] = bang_anh_xa[ma_tran_kenh[i, j]]

    return anh_da_can_bang

# anh xam 
anh_xam_can_bang = can_bang_histogram_thu_cong(anh_xam)
his_anh_xam_can_bang = tinh_histogram_thu_cong(anh_xam_can_bang)

cv2.imshow("Anh Xam Da Can Bang", anh_xam_can_bang)
cv2.waitKey(0)

plt.figure(figsize=(10, 5))
plt.plot(his_anh_xam_can_bang, color="purple", label="Ảnh xám đã cân bằng")
plt.title("Biểu đồ Histogram ảnh xám sau cân bằng")
plt.xlabel("Mức xám")
plt.ylabel("Số pixel")
plt.legend()
plt.grid(True)
plt.show()


anh_rgb_can_bang = np.zeros((chieu_cao, chieu_rong, 3), dtype=np.uint8)
xanh_duong_cb = can_bang_histogram_thu_cong(kenh_xanh_duong)
xanh_la_cb = can_bang_histogram_thu_cong(kenh_xanh_la)
do_cb = can_bang_histogram_thu_cong(kenh_do)

for i in range(chieu_cao):
    for j in range(chieu_rong):
        anh_rgb_can_bang[i, j, 0] = xanh_duong_cb[i, j]
        anh_rgb_can_bang[i, j, 1] = xanh_la_cb[i, j]
        anh_rgb_can_bang[i, j, 2] = do_cb[i, j]

kenh_y = np.zeros((chieu_cao, chieu_rong), dtype=np.uint8)
kenh_cr = np.zeros((chieu_cao, chieu_rong), dtype=np.float32)
kenh_cb = np.zeros((chieu_cao, chieu_rong), dtype=np.float32)

for i in range(chieu_cao):
    for j in range(chieu_rong):
        b_val = float(anh_mau_goc[i, j, 0])
        g_val = float(anh_mau_goc[i, j, 1])
        r_val = float(anh_mau_goc[i, j, 2])

        y_val = 0.299 * r_val + 0.587 * g_val + 0.114 * b_val
        cb_val = 128 + (-0.168736 * r_val - 0.331264 * g_val + 0.5 * b_val)
        cr_val = 128 + (0.5 * r_val - 0.418688 * g_val - 0.081312 * b_val)

        kenh_y[i, j] = np.uint8(np.clip(y_val, 0, 255))
        kenh_cb[i, j] = cb_val
        kenh_cr[i, j] = cr_val

kenh_y_can_bang = can_bang_histogram_thu_cong(kenh_y)

anh_ycrcb_can_bang = np.zeros((chieu_cao, chieu_rong, 3), dtype=np.uint8)
for i in range(chieu_cao):
    for j in range(chieu_rong):
        y_val = float(kenh_y_can_bang[i, j])
        cb_val = kenh_cb[i, j]
        cr_val = kenh_cr[i, j]

        r_val = y_val + 1.402 * (cr_val - 128)
        g_val = (
            y_val
            - 0.344136 * (cb_val - 128)
            - 0.714136 * (cr_val - 128)
        )
        b_val = y_val + 1.772 * (cb_val - 128)

        anh_ycrcb_can_bang[i, j, 0] = np.uint8(np.clip(b_val, 0, 255))
        anh_ycrcb_can_bang[i, j, 1] = np.uint8(np.clip(g_val, 0, 255))
        anh_ycrcb_can_bang[i, j, 2] = np.uint8(np.clip(r_val, 0, 255))


cv2.imshow("Anh Mau Goc", anh_mau_goc)
cv2.imshow("Can Bang Mau - RGB", anh_rgb_can_bang)
cv2.imshow("Can Bang Mau - YCrCb", anh_ycrcb_can_bang)

cv2.waitKey(0)
cv2.destroyAllWindows()