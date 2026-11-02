import cv2
import numpy as np
import math
img = cv2.imread('file_anh/Screenshot 2026-10-03 at 19.00.53.png')
cao , ngang , c = img.shape
img_amban = np.zeros((cao , ngang , 3) , dtype = np.uint8)
img_loga = np.zeros((cao , ngang , 3) , dtype = np.uint8)
img_hammu = np.zeros((cao, ngang , 3) , dtype = np.uint8)

#hang so bien doi
c_log = 255.0 / math.log(1 + 255)
gamma = 0.5
for y in range(cao):
    for x in range(ngang):
           for z in range(c):
                r = max(0.0 , float(img[y, x, z]))
                s_amban = 255 - r
                img_amban[y , x ,z] = np.uint8(s_amban)

                s_loga = c_log * math.log(1 + r)
                s_loga = min(max(s_loga , 0) , 255)
                img_loga[y , x , z] = np.uint8(s_loga)

                #255.0 * ((r / 255.0) ** gamma)
                s_hammu = 255.0 * ((r/255.0) ** gamma)
                s_hammu = min(max(s_hammu , 0) , 255)
                img_hammu[y ,x , z] = np.uint8(s_hammu)

cv2.imshow('anh am ban' , img_amban)
cv2.imshow('anh bien doi loga' , img_loga)
cv2.imshow('anh bien doi ham mu ' , img_hammu)

# cho thoat 
cv2.waitKey(0) 
cv2.destroyAllWindows()        