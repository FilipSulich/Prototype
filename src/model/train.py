from model.model import SiameseCNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm  
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import TLESSDataset


def train_model(train_json='train_pairs.json', val_json='val_pairs.json', epochs=20, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = SiameseCNN(pretrained=True, freeze_backbone=True).to(device)

    train_dataset = TLESSDataset(train_json, crop_objects=True)
    val_dataset = TLESSDataset(val_json, crop_objects=True)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    criterion_similarity = nn.BCELoss()
    criterion_angle = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for ref_img, query_img, match_label, angle_diff in train_loader_tqdm:
            ref_img = ref_img.to(device)
            query_img = query_img.to(device)
            match_label = match_label.to(device).unsqueeze(1)
            angle_diff = angle_diff.to(device).unsqueeze(1)

            optimizer.zero_grad()

            similarity, pred_angle = model(ref_img, query_img)

            loss_sim = criterion_similarity(similarity, match_label)
            loss_angle = criterion_angle(pred_angle, angle_diff)
            loss = loss_sim + 0.5 * loss_angle

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loader_tqdm.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for ref_img, query_img, match_label, angle_diff in val_loader_tqdm:
                ref_img = ref_img.to(device)
                query_img = query_img.to(device)
                match_label = match_label.to(device).unsqueeze(1)
                angle_diff = angle_diff.to(device).unsqueeze(1)

                similarity, pred_angle = model(ref_img, query_img)

                loss_sim = criterion_similarity(similarity, match_label)
                loss_angle = criterion_angle(pred_angle, angle_diff)
                loss = loss_sim + 0.5 * loss_angle

                val_loss += loss.item()
                val_loader_tqdm.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1} complete - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'checkpoints/best_model.pth')
            print(f"Best model saved (Val Loss: {best_val_loss:.4f})")

    torch.save(model.state_dict(), 'checkpoints/final_model.pth')
    print("training is complete")

    return model


if __name__ == '__main__':
    model = train_model(epochs=20, batch_size=16)
