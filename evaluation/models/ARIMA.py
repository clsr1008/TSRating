import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import torch.nn as nn
import torch

class Model(nn.Module):
    """
    ARIMA Model for Time Series Forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.order = configs.arima_order if hasattr(configs, 'arima_order') else (5, 1, 0)

    def arima_forecast(self, series):
        """
        Fit ARIMA model and make forecasts.
        :param series: Input time series (numpy array)
        :return: Forecasted values
        """
        try:
            model = ARIMA(series, order=self.order)
            fitted_model = model.fit()
            forecast = fitted_model.forecast(steps=self.pred_len)
            return forecast
        except Exception as e:
            print(f"ARIMA model fitting error: {e}")
            return np.zeros(self.pred_len)

    def forward(self, x_enc, x_mark_enc, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward function compatible with other models
        """
        batch_size = x_enc.shape[0]
        output = []

        for i in range(batch_size):
            series = x_enc[i, :, 0].detach().cpu().numpy()  # Assume target series is in the first feature
            forecast = self.arima_forecast(series)
            output.append(forecast)

        output = np.array(output)

        # Reshape to [B, pred_len, 1]
        output = output.reshape(batch_size, self.pred_len, 1)

        return torch.tensor(output, dtype=torch.float32)

