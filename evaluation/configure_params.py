import numpy as np
from models.ARIMA import Model
from dataset import data_provider

def configure_model_params(args):
    """ You can change model parameters here """
    if args.model in ['TimesNet', 'LightTS', 'DLinear', 'LSTM', 'CNN', 'Linear']:
        args.dff = 512
        args.d_model = 256
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.top_k = 5
    if args.model == 'Informer':
        args.e_layers = 3
        args.batch_size = 16
        args.d_model = 128
        args.d_ff = 256
        args.top_k = 3
        args.learning_rate = 0.001
        args.train_epochs = 100
        args.patience = 10
    if args.model == 'Nonstationary_Transformer' or 'Autoformer':
        args.dff = 512
        args.d_model = 256
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.train_epochs = 3
    if args.model == 'PatchTST':
        args.dff = 512
        args.d_model = 256
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.batch_size = 16
    if args.model == 'TimeMixer':
        args.dff = 32
        args.d_model = 16
        args.e_layers = 3
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.batch_size = 32
        args.learning_rate = 0.01
        args.train_epochs = 20
        args.patience = 10
        args.down_sampling_layers = 3
        args.down_sampling_method = 'avg'
        args.down_sampling_window = 2
        args.label_len = 0
    if args.model == 'iTransformer':
        args.d_model = 512
        args.d_ff = 512
        args.batch_size = 16
        args.learning_rate = 0.0005
        args.enc_in = 1
        args.dec_in = 1
        args.c_out = 1
        args.e_layers = 3
        args.d_layers = 1
        args.factor = 3


def print_experiment_results(score_keys, proportions, results, num_iterations):
    """
    Print the experimental results for each round.
    :param score_keys: List of score metric names.
    :param proportions: Different data proportions or models.
    :param results: Dictionary containing the experimental results.
    :param num_iterations: Total number of iterations in the experiment.
    """
    print("\nexperiment results")

    header = ["Score Key"] + proportions

    column_width = max(len(str(item)) for item in header)
    for i in range(num_iterations):
        print("\n" + f"Iteration {i + 1}")
        print("\t".join([f"{item:<{column_width}}" for item in header]))
        for score_key in score_keys:
            row = [score_key]
            for proportion in proportions:
                rmse_list = results[score_key][proportion]
                value = f"{rmse_list[i]:.3f}" if i < len(rmse_list) else "N/A"
                row.append(value)
            print("\t".join([f"{item:<{column_width}}" for item in row]))

    print("\n" + "average results")
    print("\t".join([f"{item:<{column_width}}" for item in header]))
    for score_key in score_keys:
        row = [score_key]
        for proportion in proportions:
            rmse_list = results[score_key][proportion]
            if rmse_list:
                average_rmse = sum(rmse_list) / len(rmse_list)
                row.append(f"{average_rmse:.3f}")
            else:
                row.append("N/A")
        print("\t".join([f"{item:<{column_width}}" for item in row]))

    print("\n" + "smallest results")
    print("\t".join([f"{item:<{column_width}}" for item in header]))
    for score_key in score_keys:
        row = [score_key]
        for proportion in proportions:
            rmse_list = results[score_key][proportion]
            if rmse_list:
                min_rmse = min(rmse_list)
                row.append(f"{min_rmse:.3f}")
            else:
                row.append("N/A")
        print("\t".join([f"{item:<{column_width}}" for item in row]))



def arima_training_and_testing(args, setting):
    """
    Function to train and test ARIMA model using the entire training set.
    :param args: Configuration arguments
    :param setting: Experiment setting string
    :param rmse_list: List to store RMSE values
    """
    print(f">>>>>>> Start ARIMA model training and testing: {setting} >>>>>>>>>>>>>>>>>")
    data_set, data_loader = data_provider(args, flag="train")
    arima_model = Model(args)

    listrmse = []
    for i, data_point in enumerate(data_set):
        x_enc, y_true, _, _ = data_point
        y_true = y_true[len(y_true) // 2:]

        forecast = arima_model.arima_forecast(x_enc[:, 0])

        rmse = np.sqrt(((forecast - y_true[:, 0]) ** 2).mean())
        listrmse.append(rmse)

    rmse_avg = np.mean(listrmse)

    return rmse_avg



