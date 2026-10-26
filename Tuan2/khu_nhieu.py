import cv2
import numpy as np

anh_mau = cv2.imread('file_anh/Screenshot 2026-10-03 at 19.00.53.png')
# cau 1 : tach rieng 3 kenh mau va chuyenn doi anh mau thanh anh xam theo cong thuc
b , g , r = cv2.split(anh_mau)
ngang , doc = b.shape
anh_xam = np.zeros((ngang , doc) , dtype = np.uint8)
for y in range(ngang):
    for x in range(doc):
        b_gray = b[y , x]
        g_gray = g[y , x]
        r_gray = r[y , x]

        solve_gray = 0.299 * r_gray + 0.587 * g_gray + 0.114 * b_gray
        anh_xam[y , x] = np.uint8(solve_gray)
cv2.imshow('anh sau khi chuyen doi' , anh_xam)

#bien doi do sang, do toi , do tuong phan cua anh theo tung kenh mau b g r rieng
cv2.waitKey(0) 
cv2.destroyAllWindows() #dong tat ca cua so 