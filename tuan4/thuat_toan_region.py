import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def region_growing(gray_img, seed, threshold=15):
    h, w = gray_img.shape
    segmented = np.zeros((h, w), dtype=np.uint8)
    visited = np.zeros((h, w), dtype=bool)
    
    #khoi tao hang doi 
    queue = deque()
    seed_x, seed_y = seed
    
    queue.append((seed_x, seed_y))
    visited[seed_y, seed_x] = True
    
    seed_val = float(gray_img[seed_y, seed_x])
    
    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]
    
    # kha giong thuat toan BFS
    while queue:
        cx, cy = queue.popleft()
        
        segmented[cy, cx] = 255
        
        for dx, dy in neighbors:
            nx, ny = cx + dx, cy + dy
            
           #vuot bien
            if 0 <= nx < w and 0 <= ny < h:
                # loang den vung chua duyet
                if not visited[ny, nx]:
                    current_val = float(gray_img[ny, nx])
                    
                    #ngưỡng Threshold
                    if abs(current_val - seed_val) <= threshold:
                        visited[ny, nx] = True
                        queue.append((nx, ny))
                        
    return segmented

if __name__ == "__main__":
    img = cv2.imread('file_anh/ISO_C++_Logo.jpg', cv2.IMREAD_GRAYSCALE)

        
    seed_point = (100, 100)
    result = region_growing(img, seed_point, threshold=20)
    
    plt.subplot(1, 2, 1); plt.imshow(img, cmap='gray'); plt.title("Anh Xam Goc")
    plt.subplot(1, 2, 2); plt.imshow(result, cmap='gray'); plt.title("Region")
    plt.show()