from model.model import SiameseCNN
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import sys
from pathlib import Path
import json
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset import TLESSDataset


def train_model(train_json='train_pairs.json', val_json='val_pairs.json', epochs=10, batch_size=16, lr=1e-4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = SiameseCNN(pretrained=True, freeze_backbone=False).to(device)  # unfreeze for fine-tuning

    train_dataset = TLESSDataset(train_json, crop_objects=True, augment=True)  # augmentation ON for training
    val_dataset = TLESSDataset(val_json, crop_objects=True, augment=False)  # augmentation OFF for validation


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True, persistent_workers=True, prefetch_factor=2) 
    
    pos_count = train_dataset.get_pos_count() # number of positive pairs(match)
    neg_count = train_dataset.get_neg_count() # number of negative pairs (no match)

    pos_weight = torch.tensor([neg_count / pos_count]).to(device)

    criterion_similarity = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_angle = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    os.makedirs('checkpoints', exist_ok=True)
    best_val_loss = float('inf')
    patience = 7
    patience_counter = 0

    metrics_history = {
        'train_loss': [],
        'train_loss_sim': [],
        'train_loss_angle': [],
        'train_accuracy': [],
        'train_angle_mae': [],
        'val_loss': [],
        'val_loss_sim': [],
        'val_loss_angle': [],
        'val_accuracy': [],
        'val_angle_mae': [],
        'learning_rate': []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_loss_sim_total = 0.0
        train_loss_angle_total = 0.0
        train_correct = 0
        train_total = 0
        train_angle_errors = []

        print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

        train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False)
        for ref_img, query_img, match_label, angle_diff in train_loader_tqdm:
            ref_img = ref_img.to(device)
            query_img = query_img.to(device)
            match_label = match_label.to(device).unsqueeze(1)
            angle_diff = angle_diff.to(device).unsqueeze(1) / 180.0

            optimizer.zero_grad()

            similarity, pred_angle = model(ref_img, query_img)

            loss_sim = criterion_similarity(similarity, match_label)
            loss_angle = criterion_angle(pred_angle, angle_diff)
            loss = loss_sim + 3.0 * loss_angle  # reduced from 5.0 for better balance

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_loss_sim_total += loss_sim.item()
            train_loss_angle_total += loss_angle.item()

            preds = (torch.sigmoid(similarity) > 0.5).float()
            train_correct += (preds == match_label).sum().item()
            train_total += match_label.size(0)

            angle_error = torch.abs(pred_angle - angle_diff) * 180.0
            train_angle_errors.extend(angle_error.detach().cpu().numpy())

            train_loader_tqdm.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        val_loss_sim_total = 0.0
        val_loss_angle_total = 0.0
        val_correct = 0
        val_total = 0
        val_angle_errors = []

        val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
            for ref_img, query_img, match_label, angle_diff in val_loader_tqdm:
                ref_img = ref_img.to(device)
                query_img = query_img.to(device)
                match_label = match_label.to(device).unsqueeze(1)
                angle_diff = angle_diff.to(device).unsqueeze(1) / 180.0

                similarity, pred_angle = model(ref_img, query_img)

                loss_sim = criterion_similarity(similarity, match_label)
                loss_angle = criterion_angle(pred_angle, angle_diff)
                loss = loss_sim + 3.0 * loss_angle  # reduced from 5.0 for better balance

                val_loss += loss.item()
                val_loss_sim_total += loss_sim.item()
                val_loss_angle_total += loss_angle.item()

                preds = (torch.sigmoid(similarity) > 0.5).float()
                val_correct += (preds == match_label).sum().item()
                val_total += match_label.size(0)

                angle_error = torch.abs(pred_angle - angle_diff) * 180.0
                val_angle_errors.extend(angle_error.cpu().numpy())

                val_loader_tqdm.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100.0 * train_correct / train_total
        val_acc = 100.0 * val_correct / val_total
        train_angle_mae = sum(train_angle_errors) / len(train_angle_errors)
        val_angle_mae = sum(val_angle_errors) / len(val_angle_errors)

        metrics_history['train_loss'].append(avg_train_loss)
        metrics_history['train_loss_sim'].append(train_loss_sim_total / len(train_loader))
        metrics_history['train_loss_angle'].append(train_loss_angle_total / len(train_loader))
        metrics_history['train_accuracy'].append(train_acc)
        metrics_history['train_angle_mae'].append(train_angle_mae)
        metrics_history['val_loss'].append(avg_val_loss)
        metrics_history['val_loss_sim'].append(val_loss_sim_total / len(val_loader))
        metrics_history['val_loss_angle'].append(val_loss_angle_total / len(val_loader))
        metrics_history['val_accuracy'].append(val_acc)
        metrics_history['val_angle_mae'].append(val_angle_mae)
        metrics_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch + 1} complete - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"  Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        print(f"  Train Angle MAE: {float(train_angle_mae):.2f}°, Val Angle MAE: {float(val_angle_mae):.2f}°")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch
            }, 'checkpoints/best_checkpoint.pth')
            print(f"Best model saved (Val Loss: {best_val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    torch.save(model.state_dict(), 'checkpoints/final_model.pth')

    with open('checkpoints/training_metrics.json', 'w') as f:
        json.dump(metrics_history, f, indent=2)

    print("training complete!")

    return model




model = train_model(epochs=10, batch_size=16)
