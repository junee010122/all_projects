import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import torch.nn.functional as F
import torch

plt.style.use("ggplot")
def plot_image(x, y, y_pred, output_seq, img_dim, input_seq):
    y_pred_reshaped = y_pred.view(-1, output_seq, int(np.sqrt(img_dim[0])), int(np.sqrt(img_dim[1])))
    y_reshaped = y.view(-1, output_seq, int(np.sqrt(img_dim[0])), int(np.sqrt(img_dim[1])))
    x_reshaped = x.view(-1, output_seq+input_seq, int(np.sqrt(img_dim[0])), int(np.sqrt(img_dim[1])))

    from IPython import embed
     
    #mse = F.mse_loss(y_pred_reshaped, y_reshaped, reduction='mean')

    total_cols = max(input_seq+output_seq, output_seq)
    fig, axes = plt.subplots(3, total_cols, figsize=(2 * total_cols, 6))

    for i in range(input_seq+output_seq):

        axes[0, i].imshow(x_reshaped[0, i].cpu().detach().numpy(), cmap='gray')
        axes[0, i].set_title(f'Input Image {i + 1}')
        axes[0, i].axis('off')

    diff = input_seq+output_seq - output_seq
    for i in range(diff, total_cols):
        axes[1, i].imshow(y_reshaped[0, i - diff].cpu().detach().numpy(), cmap='gray')
        axes[1, i].set_title(f'True Image {i - diff + 1}')
        axes[1, i].axis('off')

        axes[2, i].imshow(y_pred_reshaped[0, i - diff].cpu().detach().numpy(), cmap='gray')
        axes[2, i].set_title(f'Predicted Image {i - diff + 1}')
        axes[2, i].axis('off')
    for i in range(diff):
        axes[1, i].axis('off')
        axes[2, i].axis('off')

    plt.tight_layout()
    #embed()
    #plt.show()


def plot_image2(y, y_pred, output_seq, img_dim):

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

