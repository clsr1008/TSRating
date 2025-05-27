import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn.functional as F
import copy

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
        embeddings_dict: Dict[int, Tensor], the feature embedding for each sample
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
    def __init__(self, input_dim, hidden_dim=256, num_layers=3):
        super(ScoreModel, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers - 2)
        ])
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.activation = nn.ReLU()
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers - 2)
        ])

    def forward(self, x):
        x = self.activation(self.input_layer(x))
        for layer, norm in zip(self.hidden_layers, self.norm_layers):
            residual = x
            x = self.activation(norm(layer(x)))
            x = x + residual
        x = self.output_layer(x)
        return x.squeeze(-1)



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


def few_shot_finetune(model, dataset, few_shot_ratio, adaptation_steps, adaptation_lr, device='cuda'):
    """
    Split the dataset into support/test subsets and perform few-shot fine-tuning on the support set.
    Returns the fine-tuned model and the test set.
    """
    model = copy.deepcopy(model).to(device)
    loss_fn = BradleyTerryLoss()

    # === Step 1: Split support/test subsets ===
    total_len = len(dataset)
    few_shot_len = max(1, int(total_len * few_shot_ratio))
    test_len = total_len - few_shot_len
    support_set, test_set = random_split(dataset, [few_shot_len, test_len])

    # === Step 2: Few-shot fine-tuning ===
    support_loader = DataLoader(support_set, batch_size=1, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=adaptation_lr)
    model.train()

    for step in range(adaptation_steps):
        for emb_a, emb_b, p in support_loader:
            emb_a = emb_a.to(device)
            emb_b = emb_b.to(device)
            p = p.to(device)

            scores_a = model(emb_a)
            scores_b = model(emb_b)

            loss = loss_fn(scores_a, scores_b, p)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model, test_set


def evaluate_model(model, dataset, batch_size, device='cuda', finetune=True,
                   few_shot_ratio=0.05, adaptation_steps=10, adaptation_lr=5e-3):

    if finetune:
        model, dataset = few_shot_finetune(
            model, dataset,
            few_shot_ratio=few_shot_ratio,
            adaptation_steps=adaptation_steps,
            adaptation_lr=adaptation_lr,
            device=device
        )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():
        for embedding_a, embedding_b, p_b_greater_a in dataloader:
            embedding_a = embedding_a.to(device)
            embedding_b = embedding_b.to(device)
            p_b_greater_a = p_b_greater_a.to(device)
            # Compute model scores
            scores_a = model(embedding_a)
            scores_b = model(embedding_b)

            # Predict which sample has a higher score
            predicted_class = (scores_b > scores_a).float()

            # Convert p_b_greater_a into binary class labels
            true_class = (p_b_greater_a > 0.5).float()

            # Check if the prediction is correct
            correct_predictions += (predicted_class == true_class).sum().item()
            total_samples += len(p_b_greater_a)

    # Compute accuracy
    accuracy = correct_predictions / total_samples
    print(f"Evaluation Accuracy: {accuracy:.4f}")
    return accuracy

