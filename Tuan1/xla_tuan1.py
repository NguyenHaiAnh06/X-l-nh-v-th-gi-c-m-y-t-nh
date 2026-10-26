import cv2
import numpy as np

anhxl = cv2.imread('file_anh/Screenshot 2026-01-17 at 14.42.34.png')
cv2.imshow('test file anh',anhxl)
anh_xam = cv2.cvtColor(anhxl,cv2.COLOR_BGR2GRAY)
cv2.imshow('chuyen doi anh xam ',anh_xam)
rong, dai = anh_xam.shape

# cau 1 cho anh chen 1 o trang va
cau1 = anh_xam.copy()
for x in range(50 , 150):
    for y in range(100 , 170):
        cau1[x,y] = 255

cv2.imshow("anh cau 1 " , cau1)

# cau 2 cho anh ve diem anh den
cau2 = anh_xam.copy()
for x in range(50 , 150):
    for y in range(100 , 170):
        cau2[x,y] = 0

cv2.imshow('cau 2 xu ly anh den' , cau2)

#cau 3 chen gia tri la 128 vao anh 
cau3 = anh_xam.copy()
for x in range(50 , 150):
    for y in range(100 , 170):
        cau3[x, y] = 128
cv2.imshow('anh cau 3',cau3)

# cau 4 lam theo yeu cau cac hinh ve 
cau4 = anh_xam.copy()
xmin , xmax = 140 , 300
ymin, ymax = 170 , 330
duong_vien = 25 
for x in range(xmin,xmax + 1):
    for y in range(ymin , ymax + 1):
        if x < xmin + duong_vien or x > xmax - duong_vien or y < y + ymin or y > y - ymax:
            cau4[x , y] = 255
cv2.imshow('anh dau tien cua cau 4',cau4)

#cau 4 anh 2 x = 130, 351 ; y = 180, 261
# dai ngang = 190, 271 ; y = 120, 321
c4_anh2 = anh_xam.copy()
for x in range(180 , 261):
    for y in range(130, 351):
        c4_anh2[x , y] = 255
for x in range(120 , 321):
    for y in range(190 , 271):
        c4_anh2[x ,y] = 255
cv2.imshow('anh hinh chu thap' , c4_anh2)


# // an phim bat ky
cv2.waitKey(0) 
cv2.destroyAllWindows() #dong tat ca cua so 