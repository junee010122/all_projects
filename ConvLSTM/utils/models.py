import torch
import torch.nn as nn
import lightning as L
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from utils.plots import plot_image
from IPython import embed


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        self.device = 'cpu'
        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)


    def forward(self, input_tensor, cur_state):

        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)
        embed()
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.device))

class RecurrentConvLSTM(L.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters(params)

        self.conv_lstm1 = ConvLSTMCell(1, 32, kernel_size=3)
        self.conv_lstm2 = ConvLSTMCell(32, 32, kernel_size=3)
        self.conv_lstm3 = ConvLSTMCell(32, 32, kernel_size=3)
        self.conv_lstm4 = ConvLSTMCell(32, 4, kernel_size=3)

        self.mse = MeanSquaredError()
        self.mae = MeanAbsoluteError()
        #self.ssim = SSIM(data_range=1.0, channel_dim=1)

    def forward(self, x):
        batch_size, seq_len, height_width = x.size()
        height = int(height_width ** 0.5)  # Assuming square images
        width = height
        channels = 1  # Grayscale image

        x = x.view(batch_size, seq_len, channels, height, width)

        h1, c1 = self.conv_lstm1.init_hidden(batch_size, (height, width))
        h2, c2 = self.conv_lstm2.init_hidden(batch_size, (height, width))
        h3, c3 = self.conv_lstm3.init_hidden(batch_size, (height, width))
        h4, c4 = self.conv_lstm4.init_hidden(batch_size, (height, width))
        
        


        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :, :]
            h1, c1 = self.conv_lstm1(x_t, (h1, c1))
            h2, c2 = self.conv_lstm2(h1, (h2, c2))
            h3, c3 = self.conv_lstm3(h2, (h3, c3))
            h4, c4 = self.conv_lstm4(h3, (h4, c4))
            outputs.append(h4.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs[:, -self.hparams['dataset']['output_seq']:]


    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred_reshaped = y_pred.mean(dim=2).view(1, 5, -1)
        loss = self.mse(y_pred_reshaped, y)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        y_pred_reshaped = y_pred.mean(dim=2).view(1, 5, -1)
        loss = self.mse(y_pred_reshaped, y)
        # Plotting and saving images
        #plot_image(x, y, y_pred_reshaped, 5, (self.output_size, self.output_size), self.input_seq, self.path_save) 

        # Logging the validation loss and metrics
        self.log('valid_loss', loss, batch_size=1, on_step=True,
                 on_epoch=True, sync_dist=True)

    # Additional metrics
        measures = {"valid_mse": self.mse, "valid_mae": self.mae, "valid_emd": self.emd}
        for key, metric in measures.items():
            score = metric(y_pred_reshaped, y)
            self.log(key, score, batch_size=1, on_step=True,
                    on_epoch=True, sync_dist=True)

        return loss

    def emd(self, pred, target):
        # Compute EMD as the cumulative difference between the normalized histograms
        pred = pred / pred.sum(dim=-1, keepdim=True)
        target = target / target.sum(dim=-1, keepdim=True)
        cdf_pred = torch.cumsum(pred, dim=-1)
        cdf_target = torch.cumsum(target, dim=-1)
        emd_loss = torch.mean(torch.abs(cdf_pred - cdf_target))
        return emd_loss


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams['arch']['learning_rate'])


