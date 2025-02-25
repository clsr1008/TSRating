import numpy as np
import os
import shutil
from exp import Exp_Forecast, Exp_Classification
import torch
import random
import argparse
from configure_params import configure_model_params, print_experiment_results


def main(args):
    args.task_name = 'long_term_forecast'
    args.file_path = "../datasets/traffic/traffic.csv"
    args.data = 'custom'
    args.seq_len = 96  # 输入序列长度
    args.label_len = 32  # 标签长度
    args.pred_len = 32  # 预测长度
    args.start_idx = 4000
    args.end_idx = 8000
    args.feature = 'S'
    args.model_id = 'TRA_96_32'
    args.target = 'OT'
    # args.timeenc = 0
    #mape

    args.num_workers = 0 #变小有用
    args.score_file = "../middleware/traffic/annotation.jsonl"
    # args.score_key = "pattern_score"
    args.proportion = 0.5
    args.temperature = 1.0 #升序为0.0

    args.model = 'Nonstationary_Transformer'
    configure_model_params(args)
    args.itr = 1

    # score_keys = ['random','DataOob','DataShapley','KNNShapley','mix','trend_score','frequency_score','amplitude_score','pattern_score']
    score_keys = ['random', 'DataOob', 'DataShapley', 'KNNShapley', 'TimeInf', 'mix']
    models = ['Linear','CNN','iTransformer','PatchTST','TimeMixer']
    # proportions = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55]
    #升序改dataset文件

    results = {key: {model: [] for model in models} for key in score_keys}  # 初始化结果存储结构


    if args.task_name == 'long_term_forecast' or args.task_name == 'short_term_forecast':
        Exp = Exp_Forecast
    elif args.task_name == 'classification':
        Exp = Exp_Classification

    for ii in range(args.itr):
        print(f"======== Experiment iteration {ii + 1} / {args.itr} ========")
        for score_key in score_keys:
            args.score_key = score_key
            for model in models:
                # args.proportion = proportion
                args.model = model #这里记得切换
                configure_model_params(args)

                # 设置实验名称，记录参数
                setting = '{}_{}_{}_ft{}_{}_{}_{}_{}'.format(
                    args.task_name,
                    args.model_id,
                    args.model,
                    args.features,
                    args.score_key,
                    args.proportion,
                    args.des, ii
                )

                # if args.model == "ARIMA":
                #     # 调用 ARIMA 封装好的训练和测试函数
                #     rmse = arima_training_and_testing(args, setting)
                # else:
                print(f">>>>>>> Start training: {setting} >>>>>>>>>>>>>>>>>")

                exp = Exp(args)  # 初始化实验类
                exp.train(setting)  # 训练
                result = exp.test(setting)  # 测试

                # 更新当前结果
                if score_key not in results:
                    results[score_key] = {}
                if model not in results[score_key]:
                    results[score_key][model] = []
                results[score_key][model].append(result)

                # 清空 GPU 缓存
                if args.gpu_type == 'mps':
                    torch.backends.mps.empty_cache()
                elif args.gpu_type == 'cuda':
                    torch.cuda.empty_cache()

    print_experiment_results(score_keys, models, results, args.itr)

    # 删除 ./checkpoints/ 文件夹
    checkpoints_path = "./checkpoints/"
    if os.path.exists(checkpoints_path):
        shutil.rmtree(checkpoints_path)
        print(f"已删除文件夹: {checkpoints_path}")
    else:
        print(f"文件夹不存在: {checkpoints_path}")

def set_random_seed(seed: int = 2021):
    """
    设置随机种子以确保实验结果的可复现性。
    :param seed: 要设置的随机种子值，默认为 2021
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == '__main__':
    set_random_seed(2021)

    parser = argparse.ArgumentParser(description='Long Term Forecast Experiment')
    # basic config
    parser.add_argument('--task_name', type=str,  default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    # parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str,  default='test', help='model id')
    parser.add_argument('--model', type=str, default='Autoformer',
                        help='model name, options: [Autoformer, Transformer, TimesNet]')

    parser.add_argument('--start_idx', type=int,  default=4000, help='start_idx')
    parser.add_argument('--end_idx', type=int,  default=8000, help='end_idx')
    parser.add_argument('--scale', type=bool, default=True, help='是否对数据进行归一化')
    parser.add_argument('--timeenc', type=int, default=1, help='时间编码的模式 (0: 无时间编码, 1: 使用时间编码)')


    parser.add_argument('--score_file', type=str, default='../middleware/electricity/annotation.jsonl',
                        help='得分文件路径')
    parser.add_argument('--score_key', type=str, default='random',
                        help='得分文件中使用的键')
    parser.add_argument('--proportion', type=float, default=0.5,
                        help='选择样本的比例 (0 到 1)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='采样的温度')

    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    # parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    # parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--file_path', type=str, default='../datasets/electricity/electricity.csv', help='data file')
    parser.add_argument('--features', type=str, default='S',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    # parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

    # # inputation task
    # parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    #
    # # anomaly detection task
    # parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')

    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--seg_len', type=int, default=96,
                        help='the length of segmen-wise iteration of SegRNN')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')

    # # metrics (dtw)
    # parser.add_argument('--use_dtw', type=bool, default=False,
    #                     help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')

    # # Augmentation
    # parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    # parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    # parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    # parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    # parser.add_argument('--permutation', default=False, action="store_true",
    #                     help="Equal Length Permutation preset augmentation")
    # parser.add_argument('--randompermutation', default=False, action="store_true",
    #                     help="Random Length Permutation preset augmentation")
    # parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    # parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    # parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    # parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    # parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    # parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    # parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    # parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    # parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    # parser.add_argument('--discdtw', default=False, action="store_true",
    #                     help="Discrimitive DTW warp preset augmentation")
    # parser.add_argument('--discsdtw', default=False, action="store_true",
    #                     help="Discrimitive shapeDTW warp preset augmentation")
    # parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")

    # TimeXer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')

    args = parser.parse_args()

    main(args)
