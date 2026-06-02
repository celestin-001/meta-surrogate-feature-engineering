import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler


class CustomFeatureSelectionPFN(nn.Module):
    def __init__(self, num_features, d_model=128, nhead=4, num_layers=4, dim_feedforward=256):
        super().__init__()
        # Matches TabPFN: 12 layers, 512 embeddings, 1024 feed-forward, 4-head
        self.x_proj = nn.Linear(num_features, d_model)
        self.y_proj = nn.Linear(1, d_model)
        self.type_emb = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            batch_first=True,
            norm_first=True,
            layer_norm_eps=1e-4
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, 1)

    def forward(self, x_context, y_context, x_query):
        device = x_context.device
        ctx_emb = self.x_proj(x_context) + self.y_proj(y_context) + self.type_emb(
            torch.zeros(1, dtype=torch.long, device=device))
        q_emb = self.x_proj(x_query) + self.type_emb(torch.ones(1, dtype=torch.long, device=device))

        sequence = torch.cat([ctx_emb, q_emb], dim=1)
        out_sequence = self.transformer(sequence)

        query_out = out_sequence[:, x_context.size(1):, :]
        return self.out_proj(query_out)


def pretrain_and_save_model(kb_path="../Metadata/pandas/Pandas_Matrix_Complete.parquet", save_path="pfn_weights.pt", epochs=50):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df = pd.read_parquet(kb_path)

    identifier_cols = ["dataset - id", "feature - name", "operator", "task_type", "model", "improvement"]
    feature_cols = [c for c in df.columns if c not in identifier_cols]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    df[feature_cols] = df[feature_cols].fillna(0.0)
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0.0)
    scaler = StandardScaler()
    df[feature_cols] = scaler.fit_transform(df[feature_cols].values)
    scaler_path = "pfn_scaler.pkl"
    joblib.dump(scaler, scaler_path)
    model = CustomFeatureSelectionPFN(num_features=len(feature_cols)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
    criterion = nn.MSELoss()

    datasets = df["dataset - id"].unique()

    model.train()
    # FIX: Simulate a batch size of 16 datasets
    accumulation_steps = 16
    for epoch in range(epochs):
        total_loss = 0
        np.random.shuffle(datasets)

        # Reset gradients at the start of the epoch
        optimizer.zero_grad()

        for idx, ds_id in enumerate(datasets):
            ds_data = df[df["dataset - id"] == ds_id]
            shuffled = ds_data.sample(frac=1.0)
            split_idx = max(1, int(len(shuffled) * 0.8))
            split_idx = min(split_idx, len(shuffled) - 1)

            ctx_data = shuffled.iloc[:split_idx]
            qry_data = shuffled.iloc[split_idx:]

            x_ctx = torch.tensor(ctx_data[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            y_ctx = torch.tensor(ctx_data["improvement"].values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(
                device)

            x_qry = torch.tensor(qry_data[feature_cols].values, dtype=torch.float32).unsqueeze(0).to(device)
            y_qry = torch.tensor(qry_data["improvement"].values, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(
                device)

            # Optional: Catch NaN tensors before forward pass
            x_ctx = torch.nan_to_num(x_ctx, nan=0.0)
            y_ctx = torch.nan_to_num(y_ctx, nan=0.0)
            x_qry = torch.nan_to_num(x_qry, nan=0.0)
            y_qry = torch.nan_to_num(y_qry, nan=0.0)

            preds = model(x_ctx, y_ctx, x_qry)
            preds_flat = preds.squeeze()
            y_qry_flat = y_qry.squeeze()

            # Calculate the loss
            loss = criterion(preds_flat, y_qry_flat)

            # FIX: Normalize the loss by accumulation steps so gradients don't explode
            loss = loss / accumulation_steps
            loss.backward()

            # Record total un-normalized loss for logging
            total_loss += (loss.item() * accumulation_steps)

            # FIX: Only update weights after every 16 datasets (or at the end of the data)
            if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(datasets)):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()
                optimizer.zero_grad()  # Reset for the next 16 datasets

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {total_loss / len(datasets):.4f}")
        # torch.save(model.state_dict(), save_path)
        # print(f"Model weights saved to {save_path}")

    # Extract and save the model parameters
    torch.save(model.state_dict(), save_path)
    print(f"Model weights saved to {save_path}")


if __name__ == "__main__":
    pretrain_and_save_model()
