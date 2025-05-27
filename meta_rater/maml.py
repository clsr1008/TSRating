import copy
import torch
import torch.nn as nn
import random
import torch.optim as optim
from torch.utils.data import DataLoader
from trainer import ScoreModel, BradleyTerryLoss
import higher

class MetaLearner:
    def __init__(self, input_dim, hidden_dim, num_layers, meta_lr=1e-3, inner_lr=0.01, inner_steps=5, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.meta_model = ScoreModel(input_dim, hidden_dim, num_layers).to(self.device)
        self.meta_optimizer = optim.SGD(self.meta_model.parameters(), lr=meta_lr)
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.loss_fn = BradleyTerryLoss()

    def clone_model(self):
        return copy.deepcopy(self.meta_model)


    def train_on_task(self, task_loader):
        self.meta_model.train()
        fmodel = copy.deepcopy(self.meta_model)
        fmodel.train()

        for step in range(self.inner_steps):
            inner_loss_sum = 0.0
            count = 0
            for emb_a, emb_b, p in task_loader:
                emb_a = emb_a.to(self.device)
                emb_b = emb_b.to(self.device)
                p = p.to(self.device)

                scores_a = fmodel(emb_a)
                scores_b = fmodel(emb_b)
                loss = self.loss_fn(scores_a, scores_b, p)
                grads = torch.autograd.grad(loss, fmodel.parameters(), create_graph=True)

                # signSGD update
                with torch.no_grad():
                    for param, grad in zip(fmodel.parameters(), grads):
                        param -= self.inner_lr * grad.sign()

                inner_loss_sum += loss.item()
                count += 1

            # print(f"[Inner Step {step + 1}] Avg Inner Loss: {inner_loss_sum / count:.4f}")
        return fmodel


    def meta_train(self, task_datasets, meta_batch_size=4, data_batch_size=16, epochs=10):
        for epoch in range(epochs):
            total_meta_loss = 0.0
            sampled_tasks = random.sample(task_datasets, meta_batch_size)

            for task in sampled_tasks:
                support_loader = DataLoader(task['support'], batch_size=data_batch_size, shuffle=True, num_workers=0)
                query_loader = DataLoader(task['query'], batch_size=data_batch_size, shuffle=True, num_workers=0)
                # Inner-loop
                fmodel = self.train_on_task(support_loader)

                # calculate meta loss
                fmodel.eval()
                meta_loss = 0.0
                for emb_a, emb_b, p in query_loader:
                    emb_a = emb_a.to(self.device)
                    emb_b = emb_b.to(self.device)
                    p = p.to(self.device)

                    scores_a = fmodel(emb_a)
                    scores_b = fmodel(emb_b)
                    loss = self.loss_fn(scores_a, scores_b, p)
                    meta_loss += loss

                self.meta_optimizer.zero_grad()
                meta_loss.backward()
                self.meta_optimizer.step()
                total_meta_loss += meta_loss.item()

            print(f"[Epoch {epoch+1}] Meta Loss: {total_meta_loss / meta_batch_size:.4f}")
