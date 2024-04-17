

def plot_images(dataset, binarize=False):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    for i in range(5):
        original_image = dataset[i][0]
        if binarize:
            original_image = Image.fromarray(np.where(np.array(original_image) > 0, 255, 0).astype('uint8'))

        plt.subplot(2, 5, i + 1)
        plt.imshow(original_image, cmap='gray')
        plt.title('Original')
        plt.axis('off')

        plt.subplot(2, 5, i + 6)
        plt.imshow(original_image, cmap='gray')
        plt.title('Binarized')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

