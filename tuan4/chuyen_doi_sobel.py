import cv2
import numpy as np
import matplotlib.pyplot as plt

img_path = 'file_anh/2018-07-04T12_04_17.738369.png' 
img_color = cv2.imread(img_path)

img_color = cv2.resize(img_color, (256, 256))
gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)


Hx_5x5 = np.array([
    [-5,  -4, 0,  4,  5],
    [-8, -10, 0, 10,  8],
    [-10,-20, 0, 20, 10],
    [-8, -10, 0, 10,  8],
    [-5,  -4, 0,  4,  5]
], dtype=np.float32)

Hy_5x5 = np.array([
    [-5,  -8,-10, -8, -5],
    [-4, -10,-20,-10, -4],
    [ 0,   0,  0,  0,  0],
    [ 4,  10, 20, 10,  4],
    [ 5,   8, 10,  8,  5]
], dtype=np.float32)

gx_5x5 = cv2.filter2D(gray, cv2.CV_64F, Hx_5x5)
gy_5x5 = cv2.filter2D(gray, cv2.CV_64F, Hy_5x5)
sobel_5x5_raw = cv2.magnitude(gx_5x5, gy_5x5)


sobel_5x5 = cv2.normalize(sobel_5x5_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)


gx_3x3 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
gy_3x3 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel_3x3 = cv2.magnitude(gx_3x3, gy_3x3)
sobel_3x3 = np.clip(sobel_3x3, 0, 255).astype(np.uint8)


L_5x5 = np.array([
    [ 0,  0, -1,  0,  0],
    [ 0, -1, -2, -1,  0],
    [-1, -2, 16, -2, -1],
    [ 0, -1, -2, -1,  0],
    [ 0,  0, -1,  0,  0]
], dtype=np.float32)


laplace_5x5 = cv2.filter2D(gray, cv2.CV_64F, L_5x5)
laplace_5x5 = np.uint8(np.absolute(laplace_5x5))


laplace_3x3 = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
laplace_3x3 = np.uint8(np.absolute(laplace_3x3))


plt.figure(figsize=(14, 8))

plt.subplot(2, 3, 1)
plt.imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
plt.title("1. Ảnh gốc ")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(sobel_3x3, cmap="gray")
plt.title("2. Sobel 3x3")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(sobel_5x5, cmap="gray")
plt.title("3. Sobel 5x5 ")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(gray, cmap="gray")
plt.title("4. Ảnh xám")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(laplace_3x3, cmap="gray")
plt.title("5. Laplace 3x3")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(laplace_5x5, cmap="gray")
plt.title("6. Laplace 5x5")
plt.axis("off")

plt.tight_layout()
plt.show()