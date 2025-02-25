import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Embed import DataEmbedding


class Model(nn.Module):
    """
    Vanilla CNN Model
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len

        # Embedding Layer
        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # CNN Encoder
        self.conv_layers = nn.Sequential(
            nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=configs.d_model * 2, out_channels=configs.d_model, kernel_size=3, padding=1),
            nn.ReLU()
        )

        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
            self.decoder_conv = nn.Sequential(
                nn.Conv1d(in_channels=configs.d_model, out_channels=configs.d_model, kernel_size=3, padding=1),
                nn.ReLU()
            )
            self.projection = nn.Linear(configs.d_model, configs.c_out)

        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out)

        elif self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = enc_out.permute(0, 2, 1)  # [B, D, L] for CNN input
        enc_out = self.conv_layers(enc_out)
        enc_out = enc_out.permute(0, 2, 1)  # Back to [B, L, D]

        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = self.decoder_conv(dec_out)
        dec_out = dec_out.permute(0, 2, 1)
        dec_out = self.projection(dec_out)

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.conv_layers(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        dec_out = self.projection(enc_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.conv_layers(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        dec_out = self.projection(enc_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        enc_out = self.enc_embedding(x_enc, None)
        enc_out = enc_out.permute(0, 2, 1)
        enc_out = self.conv_layers(enc_out)
        enc_out = enc_out.permute(0, 2, 1)
        output = self.act(enc_out)
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out
        return None
