import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


# Bradley-Terry 损失函数
class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super(BradleyTerryLoss, self).__init__()

    def forward(self, scores_a, scores_b, p_b_greater_a):
        diff = scores_b - scores_a
        log_prob = torch.log(torch.sigmoid(diff)) * p_b_greater_a + torch.log(torch.sigmoid(-diff)) * (1 - p_b_greater_a)
        return -log_prob.mean()


# 数据集定义
class PairwiseDataset(Dataset):
    def __init__(self, embeddings_dict, dataframe):
        """
        embeddings_dict: Dict[int, Tensor], 每个样本的特征
        dataframe: pd.DataFrame, 包含 block_a, block_b, comparisons_avg
        """
        self.embeddings_dict = embeddings_dict
        self.dataframe = dataframe

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        block_a = row["block_a"]
        block_b = row["block_b"]
        p_b_greater_a = row["comparisons_avg"]

        embedding_a = self.embeddings_dict[block_a]
        embedding_b = self.embeddings_dict[block_b]

        return embedding_a, embedding_b, torch.tensor(p_b_greater_a, dtype=torch.float32)


# 评分模型
class ScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        """
        参数:
        input_dim: 输入特征的维度
        hidden_dim: 每个隐藏层的维度
        num_layers: 隐藏层的数量
        """
        super(ScoreModel, self).__init__()
        layers = []

        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # 输出层
        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)  # 输出形状为 [batch_size]


def train_model(model, dataset, epochs=10, batch_size=32, lr=0.001):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = BradleyTerryLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for embedding_a, embedding_b, p_b_greater_a in dataloader:
            optimizer.zero_grad()
            scores_a = model(embedding_a)
            scores_b = model(embedding_b)
            loss = criterion(scores_a, scores_b, p_b_greater_a)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader):.4f}")
    return model


# 评估函数
def evaluate_model(model, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for embedding_a, embedding_b, p_b_greater_a in dataloader:
            # 计算模型输出分数
            scores_a = model(embedding_a)
            scores_b = model(embedding_b)

            # 根据模型分数判断哪个样本得分更高
            predicted_class = (scores_b > scores_a).float()

            # 将 p_b_greater_a 离散化为二分类标签
            true_class = (p_b_greater_a > 0.5).float()

            # 判断是否预测正确
            correct_predictions += (predicted_class == true_class).sum().item()
            total_samples += len(p_b_greater_a)

    # 计算准确率
    accuracy = correct_predictions / total_samples
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy

