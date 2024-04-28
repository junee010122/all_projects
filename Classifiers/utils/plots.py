import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def plot_images(orig, binarized):

    plt.figure(figsize=(10, 5))
    
    for i in range(5):
        original_image = orig[i]
        binarized_image = binarized[i]
        
        plt.subplot(2, 5, i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(binarized_image, cmap='gray')
        plt.title('Binarized')
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()


def plot_pca_images(X, X_pca, pca, num_images=5):

    plt.figure(figsize=(10, 5))
    for i in range(num_images):
        original_image = X[i].reshape(28, 28)
        reconstructed_image = pca.inverse_transform(X_pca[i]).reshape(28, 28)

        plt.subplot(2, num_images, i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title('PCA')
        plt.axis('off')

    plt.tight_layout()
    plt.show()
