import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import torch.nn.functional as F


plt.style.use("ggplot")

def plot_image(y, y_pred, output_seq, img_dim):

    y_pred_reshaped = y_pred.view(-1, output_seq, int(np.sqrt(img_dim[0])), int(np.sqrt(img_dim[1])))
    y_reshaped = y.view(-1, output_seq, int(np.sqrt(img_dim[0])), int(np.sqrt(img_dim[1])))
    
    # Calculate MSE for each image in the sequence and then average
    mse = F.mse_loss(y_pred_reshaped, y_reshaped, reduction='mean')
    
    # Display all images in the sequence
    fig, axes = plt.subplots(2, output_seq, figsize=(2*output_seq, 4))
    for i in range(output_seq):
        # Display ground truth image
        axes[0, i].imshow(y_reshaped[0, i].cpu().detach().numpy(), cmap='gray')
        axes[0, i].set_title(f'True Image {i+1}')
        axes[0, i].axis('off')
        
        # Display predicted image
        axes[1, i].imshow(y_pred_reshaped[0, i].cpu().detach().numpy(), cmap='gray')
        axes[1, i].set_title(f'Predicted Image {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()
    
def make_video():
    pass

def plot_eval():
    pass

