import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator


class BradleyTerryLoss(nn.Module):
    def __init__(self):
        super(BradleyTerryLoss, self).__init__()

    def forward(self, scores_a, scores_b, p_b_greater_a):
        diff = scores_b - scores_a
        log_prob = torch.log(torch.sigmoid(diff)) * p_b_greater_a + torch.log(torch.sigmoid(-diff)) * (1 - p_b_greater_a)
        return -log_prob.mean()


class PairwiseDataset(Dataset):
    def __init__(self, embeddings_dict, dataframe):
        """
            embeddings_dict: Dict[int, Tensor], features of each sample
            dataframe: pd.DataFrame, contains block_a, block_b, comparisons_avg
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


class ScoreModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2):
        """
            Parameters:
            input_dim: The dimension of the input features
            hidden_dim: The dimension of each hidden layer
            num_layers: The number of hidden layers
        """
        super(ScoreModel, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, 1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x).squeeze(-1)  # [batch_size]


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


def evaluate_model(model, dataset, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for embedding_a, embedding_b, p_b_greater_a in dataloader:
            scores_a = model(embedding_a)
            scores_b = model(embedding_b)

            predicted_class = (scores_b > scores_a).float()

            true_class = (p_b_greater_a > 0.5).float()

            correct_predictions += (predicted_class == true_class).sum().item()
            total_samples += len(p_b_greater_a)

    # calculate accuracy
    accuracy = correct_predictions / total_samples
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy

