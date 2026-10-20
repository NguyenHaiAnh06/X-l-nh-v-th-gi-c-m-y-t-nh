import cv2

img = cv2.imread('file_anh/Screenshot 2026-01-17 at 14.42.34.png')

if img is None:
    print('Không thể đọc được ảnh. Vui lòng kiểm tra lại đường dẫn!')
    exit()

cv2.imshow('Anh goc' ,img)
# b, g, r = cv2.split(img)
# cv2.imshow(img);
# cv2.imshow('1. Kenh Red (R)', r)
# cv2.imshow('1. Kenh Green (G)', g)
# cv2.imshow('1. Kenh Blue (B)', b)


# gray_avg = cv2.addWeighted(r, 1 / 3, g, 1 / 3, 0)
# gray_avg = cv2.addWeighted(gray_avg, 1.0, b, 1 / 3, 0)

# # anh xam
# gray_lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# _, img_binary = cv2.threshold(gray_lum, 127, 255, cv2.THRESH_BINARY)

# cv2.imshow('Anh xam ', gray_avg)
# cv2.imshow('Anh xam', gray_lum)
# cv2.imshow('Anh nhi phan', img_binary)



# img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# # C = 255 - R, M = 255 - G, Y = 255 - B
# c = cv2.subtract(255, r)
# m = cv2.subtract(255, g)
# y = cv2.subtract(255, b)
# img_cmy = cv2.merge([c, m, y])

# k = cv2.min(cv2.min(c, m), y)


# cv2.imshow('3.1 Mohinh HSV', img_hsv)
# cv2.imshow('3.2 Mohinh CMY', img_cmy)
# cv2.imshow('3.3 Kenh K (Den) trong CMYK', k)


cv2.waitKey(0) 
# // an phim bat ky
cv2.destroyAllWindows()