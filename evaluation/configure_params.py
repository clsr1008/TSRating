import numpy as np
from models.ARIMA import Model
from dataset import data_provider

def configure_model_params(args):
    if args.model in ['TimesNet', 'LightTS', 'DLinear', 'LSTM', 'CNN', 'Linear']:
        args.dff = 512
        args.d_model = 256
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1  # 单变量
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
        args.enc_in = 1  # 单变量
        args.dec_in = 1
        args.c_out = 1
        args.train_epochs = 3
    if args.model == 'PatchTST':
        args.dff = 512
        args.d_model = 256
        args.e_layers = 2
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1  # 单变量
        args.dec_in = 1
        args.c_out = 1
        args.batch_size = 16
    if args.model == 'TimeMixer':
        args.dff = 32
        args.d_model = 16
        args.e_layers = 3
        args.d_layers = 1
        args.factor = 3
        args.enc_in = 1  # 输入维度
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
        args.enc_in = 1  # 输入维度
        args.dec_in = 1
        args.c_out = 1
        args.e_layers = 3
        args.d_layers = 1
        args.factor = 3


def print_experiment_results(score_keys, proportions, results, num_iterations):
    """
    打印每一轮的实验结果。
    :param score_keys: 评分指标名称列表
    :param proportions: 不同数据比例
    :param results: 包含实验结果的字典
    :param num_iterations: 实验总轮数
    """
    print("\n实验结果")

    # 表头设置
    header = ["Score Key"] + proportions

    # 确定最大列宽，避免对齐混乱
    column_width = max(len(str(item)) for item in header)

    # 逐轮打印每个评分指标和数据比例下的 RMSE
    for i in range(num_iterations):
        print("\n" + f"Iteration {i + 1}")
        print("\t".join([f"{item:<{column_width}}" for item in header])) # 打印表头
        for score_key in score_keys:
            row = [score_key]
            for proportion in proportions:
                rmse_list = results[score_key][proportion]
                value = f"{rmse_list[i]:.3f}" if i < len(rmse_list) else "N/A"
                row.append(value)
            print("\t".join([f"{item:<{column_width}}" for item in row]))

    # 打印平均结果
    print("\n" + "平均结果")
    print("\t".join([f"{item:<{column_width}}" for item in header]))  # 打印表头
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

    # 打印最小结果
    print("\n最小结果")
    print("\t".join([f"{item:<{column_width}}" for item in header]))  # 打印表头
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
    # args.proportion = 0.1
    args.temperature = 2.0
    # 使用 data_provider 加载数据
    data_set, data_loader = data_provider(args, flag="train")  # 获取训练数据

    # 创建 ARIMA 模型实例
    arima_model = Model(args)  # 创建 Model 实例，传入配置参数

    listrmse = []
    # 遍历训练集中的每一条时间序列进行训练和预测
    for i, data_point in enumerate(data_set):
        # 假设 data_point 是一个四元组，前两个元素是 x_enc 和 y_true，后两个元素不使用
        x_enc, y_true, _, _ = data_point  # 只取前两个元素
        y_true = y_true[len(y_true) // 2:]

        # 使用 x_enc 前96个数据进行 ARIMA 训练，预测后32个数据
        forecast = arima_model.arima_forecast(x_enc[:, 0])  # 假设使用第一个特征进行预测

        # 计算 RMSE（预测值与真实目标值之间的差异）
        rmse = np.sqrt(((forecast - y_true[:, 0]) ** 2).mean())  # 使用目标部分进行 RMSE 计算
        listrmse.append(rmse)

    rmse_avg = np.mean(listrmse)

    return rmse_avg



