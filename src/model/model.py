from ..data_creation.dataset import TLESSDataset

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader

import os
from tqdm import tqdm
import sys
from pathlib import Path
import json
sys.path.append(str(Path(__file__).parent.parent))

class SiameseCNN(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(SiameseCNN, self).__init__()

        resnet = models.resnet18(pretrained=pretrained) # we use the ResNet18 pre-trained model as the backbone
        
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # we use all the layers from ResNet18 except the final fully connected layer - we only need the feature extractor

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False # freeze backbone parameters (so we dont change the ResNet weights)

        self.fc_similarity = nn.Sequential(
            nn.Linear(512 * 2, 256), # input size is 512*2 due to concatenation of two feature vectors
            nn.ReLU(), # activation function
            nn.Dropout(0.5), # increased dropout to prevent overfitting
            nn.Linear(256, 128), # hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1) # output logits (no sigmoid - using BCEWithLogitsLoss)
        )

        self.fc_angle = nn.Sequential(
            nn.Linear(512 * 2, 256), # input size is 512*2 due to concatenation of two feature vectors
            nn.ReLU(), # activation function
            nn.Dropout(0.5), # increased dropout to prevent overfitting
            nn.Linear(256, 128), # hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), # output layer
            nn.Tanh() # output between -1 and 1 for angle difference (normalized for more stable training)
        )

    def forward_once(self, x):
        x = self.backbone(x) # first pass through the ResNet backbone
        x = x.view(x.size(0), -1) # flatten the output tensor
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_once(img1) # feature vector from the first image (512 dimensions)
        feat2 = self.forward_once(img2) # feature vector from the second image (512 dimensions)
        combined = torch.cat([feat1, feat2], dim=1) # concatenation of the two vectors (512 *2 = 1024 dimensions)

        similarity = self.fc_similarity(combined) # similarity score between 0 and 1
        angle_diff = self.fc_angle(combined) * 180.0 # predicted angle difference (we scale it back to degrees)
    
        return similarity, angle_diff
    
    def train_model(self, train_json='train_pairs.json', val_json='val_pairs.json', epochs=10, batch_size=16, lr=1e-4):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # model = SiameseCNN(pretrained=pretrained, freeze_backbone=freeze_backbone).to(device)  # unfreeze the ResNet18 weights for fine-tuning

        model = self.to(device)

        train_dataset = TLESSDataset(train_json, crop_objects=True, augment=True)  # the augmentation is turned on for training
        val_dataset = TLESSDataset(val_json, crop_objects=True, augment=False)  # its turned off for validating


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
                loss = loss_sim + 3.0 * loss_angle  

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
                    loss = loss_sim + 3.0 * loss_angle 

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
