import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import joblib

# Load interactions
interactions_df = pd.read_csv('interactions.csv')

# Map user and item IDs to indices
user_ids = interactions_df['user_id'].unique()
item_ids = interactions_df['product_id'].unique()
user2idx = {uid: idx for idx, uid in enumerate(user_ids)}
item2idx = {iid: idx for idx, iid in enumerate(item_ids)}

# Create dataset
class InteractionsDataset(Dataset):
    def __init__(self, df, user2idx, item2idx):
        self.users = torch.tensor([user2idx[uid] for uid in df['user_id']], dtype=torch.long)
        self.items = torch.tensor([item2idx[iid] for iid in df['product_id']], dtype=torch.long)
        self.ratings = torch.tensor(df['rating'].values, dtype=torch.float)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

# Define NCF model
class NCF(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64):
        super(NCF, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=-1)
        return self.fc_layers(x).squeeze()

# Training function
def train_ncf(model, dataloader, epochs=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for user, item, rating in dataloader:
            optimizer.zero_grad()
            pred = model(user, item)
            loss = criterion(pred, rating)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader):.4f}")

# Initialize and train
dataset = InteractionsDataset(interactions_df, user2idx, item2idx)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
model = NCF(num_users=len(user_ids), num_items=len(item_ids))
train_ncf(model, dataloader)

# Save model and mappings
torch.save(model.state_dict(), 'ncf_model.pth')
joblib.dump({'user2idx': user2idx, 'item2idx': item2idx}, 'ncf_mappings.pkl')