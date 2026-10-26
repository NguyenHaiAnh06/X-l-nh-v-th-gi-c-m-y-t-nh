import cv2
import numpy as np
anh_goc = cv2.imread('file_anh/Screenshot 2026-10-03 at 19.00.53.png')
doc , ngang , kenh_mau = anh_goc.shape
# khoi tao mang chua ket qua tranh tran so
ket_qua = np.zeros((doc , ngang , 3) , dtype = np.uint8)
alpha_b, beta_b = 1.0 , 30
alpha_g , beta_g = 1.3 , -20
alpha_r , beta_r = 1.5 , 40

for y in range(doc):
    for x in range(ngang):
        val_b = alpha_b * anh_goc[y , x , 0] + beta_b
        val_b = min(max(val_b , 0) , 255)
        ket_qua[y , x , 0] = np.uint8(val_b)

        val_g = alpha_g * anh_goc[y , x , 0] + beta_g
        val_g = min(max(val_g , 0) , 255)
        ket_qua[y , x , 1] = np.uint8(val_g)

        val_r = alpha_r * anh_goc[y , x , 0] + beta_r
        val_r = min(max(val_r , 0) , 255)
        ket_qua[y , x , 2] = np.uint8(val_r)

cv2.imshow('test anh sau khi tang sang toi va tuong phan ' , ket_qua)
cv2.waitKey(0) 
cv2.destroyAllWindows() #dong tat ca cua so 