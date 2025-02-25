import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import json
import random
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
# from data_provider.m4 import M4Dataset, M4Meta
from data_preparation.uea import subsample, interpolate_missing, Normalizer, collate_fn
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
# from utils.augmentation import run_augmentation_single
class Dataset_Custom(Dataset):
    def __init__(self, file_path, flag='train', size=None, features='S',
                 target='OT', scale=True, timeenc=1, freq='h', start_idx=4000, end_idx=8000, data='custom'):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.data = data

        self.file_path = file_path
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(self.file_path, encoding='ISO-8859-1')
        df_raw = df_raw.iloc[self.start_idx:self.end_idx] #截取连续的4000个点

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.data == 'custom':
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
            df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        if self.data == 'custom':
            df_stamp = df_raw[['date']][border1:border2]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            if self.timeenc == 0:
                df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
                df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
                df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
                data_stamp = df_stamp.drop(['date'], 1).values
            elif self.timeenc == 1:
                data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
                data_stamp = data_stamp.transpose(1, 0)
        else:
            data_stamp = None

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        if self.data == 'custom':
            seq_x_mark = self.data_stamp[s_begin:s_end]
            seq_y_mark = self.data_stamp[r_begin:r_end]
        else:
            seq_x_mark = np.ones((s_end - s_begin, 4))
            seq_y_mark = np.ones((s_end - s_begin, 4))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        # print(len(self.all_IDs))

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        if file_list is None:
            data_paths = glob.glob(os.path.join(root_path, '*'))  # list of all paths
        else:
            data_paths = [os.path.join(root_path, p) for p in file_list]
        if len(data_paths) == 0:
            raise Exception('No files found using: {}'.format(os.path.join(root_path, '*')))
        if flag is not None:
            data_paths = list(filter(lambda x: re.search(flag, x), data_paths))
        input_paths = [p for p in data_paths if os.path.isfile(p) and p.endswith('.ts')]
        if len(input_paths) == 0:
            pattern='*.ts'
            raise Exception("No .ts files found using pattern: '{}'".format(pattern))

        all_df, labels_df = self.load_single(input_paths[0])  # a single file contains dataset

        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        # if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
        #     num_samples = len(self.all_IDs)
        #     num_columns = self.feature_df.shape[1]
        #     seq_len = int(self.feature_df.shape[0] / num_samples)
        #     batch_x = batch_x.reshape((1, seq_len, num_columns))
        #     batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)
        #
        #     batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)

def select_train_samples(dataset, score_file, score_key, proportion, temperature):
    """
    从 Dataset_Custom 数据集中选择样本，并返回子集数据集。

    参数:
    - dataset: Dataset_Custom 对象，数据集实例。
    - selection_mode: str，选择模式，可选 "random" 或 "score_based"。
    - score_file: str，仅当选择模式为 "score_based" 时需要，包含得分的文件路径，支持 .jsonl 格式。
    - score_key: str，仅当选择模式为 "score_based" 时需要，得分文件中对应的键名。
    - proportion: float，选择样本的比例（必填）。
    - temperature: float，控制采样多样性的温度参数（默认值为 1.0）。

    返回:
    - SubsetDataset 对象，包含筛选后的样本子集。
    """
    if proportion is None or proportion <= 0 or proportion > 1:
        raise ValueError("必须提供一个有效的 proportion（范围为 0 到 1 之间的浮点数）。")

    # 获取数据集样本总数
    total_samples = len(dataset)

    # 根据比例计算样本数量
    num_samples = int(total_samples * proportion)

    if score_key == "random":
        # 随机选择
        selected_indices = random.sample(range(total_samples), num_samples)
    elif score_key == "mix":
        if score_file is None:
            raise ValueError("当选择模式为 'mix' 时，必须提供 score_file。")

        # 加载得分文件（支持 .jsonl 格式）
        scores_list = []  # 用于存储所有以 'score' 结尾的得分
        with open(score_file, 'r') as f:
            for line in f:
                score_data = json.loads(line.strip())
                # 提取所有以 'score' 结尾的字段
                mix_scores = [value for key, value in score_data.items() if key.endswith("score")]
                if mix_scores:
                    scores_list.append(mix_scores)

        # 确保每个样本都有完整的得分维度
        if len(scores_list) != total_samples:
            raise ValueError("得分文件中的样本数量与数据集样本总数不匹配。")

        # 将所有得分维度合并为 NumPy 数组
        scores_array = np.array(scores_list)  # shape: (total_samples, num_dimensions)

        # 对每个维度的得分进行归一化（min-max 标准化）
        min_vals = np.min(scores_array, axis=0)
        max_vals = np.max(scores_array, axis=0)
        normalized_scores = (scores_array - min_vals) / (max_vals - min_vals + 1e-8)

        # 计算每个样本的平均得分
        mean_scores = np.mean(normalized_scores, axis=1)  # 平均化所有维度得分

        # 使用温度调整选择分数最高的样本
        if temperature == 0.0:
            selected_indices = np.argsort(-mean_scores)[:num_samples].tolist() #加负号是降序，不加是升序
        else:
            # Softmax 计算
            exp_scores = np.exp(mean_scores / temperature)  # 应用温度调整的 softmax 分子
            probabilities = exp_scores / np.sum(exp_scores)  # 归一化为概率分布

            # 根据概率分布进行无放回采样
            selected_indices = np.random.choice(
                range(total_samples), size=num_samples, replace=False, p=probabilities
            ).tolist()
    else:
        if score_file is None:
            raise ValueError("当选择模式为 'score_based' 时，必须提供 score_file。")

        # 加载得分文件（支持 .jsonl 格式）
        scores = []
        with open(score_file, 'r') as f:
            for line in f:
                score_data = json.loads(line.strip())
                if score_key in score_data:
                    scores.append(score_data[score_key])

        if len(scores) != total_samples:
            raise ValueError("得分文件中的样本数量与数据集样本总数不匹配。")

        scores = np.array(scores)  # 转换为 NumPy 数组
        if temperature == 0.0 or score_key in ['DataOob', 'DataShapley', 'KNNShapley', 'TimeInf']:
            selected_indices = np.argsort(-scores)[:num_samples].tolist() #加负号是降序，不加是升序
            # print(selected_indices)
        else:
            # 对 scores 进行标准化，使方差为 1
            # mean_score = np.mean(scores)
            # std_score = np.std(scores) + 1e-8  # 避免分母为 0
            # scores = (scores - mean_score) / std_score

            # Softmax 计算
            exp_scores = np.exp(scores / temperature)  # 应用温度调整的 softmax 分子
            probabilities = exp_scores / np.sum(exp_scores)  # 归一化为概率分布

            # 根据概率分布进行无放回采样
            selected_indices = np.random.choice(
                range(total_samples), size=num_samples, replace=False, p=probabilities
            ).tolist()

    # 返回子集数据集
    return Subset(dataset, selected_indices)



def data_provider(args, flag):
    if args.data in ['m4', 'custom']:
        Data = Dataset_Custom
    elif args.data == "UEA":
        Data = UEAloader
    shuffle_flag = False if (flag == 'test' or flag == 'TEST') else True
    drop_last = False
    if args.task_name == 'classification':
        drop_last = False
        data_set = Data(
            args=args,
            root_path=args.file_path,
            flag=flag,
        )
        if flag == 'TRAIN':
            data_set = select_train_samples(
                dataset=data_set,
                score_file=args.score_file,
                score_key=args.score_key,
                proportion=args.proportion,
                temperature=args.temperature
            )

        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last,
            collate_fn=lambda x: collate_fn(x, max_len=args.seq_len)
        )
        return data_set, data_loader
    else:
        data_set = Data(
            file_path=args.file_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            scale=args.scale,
            timeenc=args.timeenc,
            freq=args.freq,
            start_idx=args.start_idx,
            end_idx=args.end_idx,
            data=args.data
        )
        if flag == 'train':
            data_set = select_train_samples(
                dataset=data_set,
                score_file=args.score_file,
                score_key=args.score_key,
                proportion=args.proportion,
                temperature=args.temperature
            )
        data_loader = DataLoader(
            data_set,
            batch_size=args.batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)
        return data_set, data_loader
