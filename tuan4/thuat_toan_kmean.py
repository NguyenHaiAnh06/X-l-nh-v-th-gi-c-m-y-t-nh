import cv2
import numpy as np
import matplotlib.pyplot as plt

def kmeans_segmentation(img_rgb, k=3, max_iters=10):
    #duoi anh
    height, width, channels = img_rgb.shape
    data = img_rgb.reshape((-1, 3)).astype(np.float64)
    n_pixels = data.shape[0]
    #choice K ngau nhien 
    np.random.seed(42)
    random_indices = np.random.choice(n_pixels, k, replace=False)
    centers = data[random_indices]
    
    labels = np.zeros(n_pixels, dtype=int)
    

    for i in range(max_iters):
        # Shape: (N_pixels, K)
        distances = np.linalg.norm(data[:, np.newaxis] - centers, axis=2)
        
        # gan pixel gan nhat
        labels = np.argmin(distances, axis=1)
        
        # tinh lai 
        new_centers = np.array([
            data[labels == j].mean(axis=0) if np.sum(labels == j) > 0 else centers[j]
            for j in range(k)
        ])
        
        #stop
        if np.allclose(centers, new_centers):
            print(f"Hội tụ tại vòng lặp thứ {i + 1}")
            break
            
        centers = new_centers
        
    segmented_data = centers[labels].astype(np.uint8)
    segmented_img = segmented_data.reshape((height, width, channels))
    
    return segmented_img

if __name__ == "__main__":
    img = cv2.imread('file_anh/ISO_C++_Logo.jpg')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = kmeans_segmentation(img_rgb, k=3, max_iters=15)
    
    plt.subplot(1, 2, 1); plt.imshow(img_rgb); plt.title("Anh Goc")
    plt.subplot(1, 2, 2); plt.imshow(result); plt.title("K means")
    plt.show()