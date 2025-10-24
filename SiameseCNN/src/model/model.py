from ..data_creation.dataset import TLESSDataset
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
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
    def __init__(self, freeze_backbone=True):
        """
        Initialize the Siamese CNN model with a ResNet18 backbone.
        Args:
            freeze_backbone (bool): Whether to freeze the ResNet8 backbone weights during training.
        """
        super(SiameseCNN, self).__init__()

        resnet = models.resnet18(weights='IMAGENET1K_V1') # we use the ResNet18 pre-trained model as the backbone
        print(resnet)
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
            nn.Linear(128, 1) # output logits for similarity score
        )

        self.fc_angle = nn.Sequential(
            nn.Linear(512 * 2, 256), # input size is 512*2 due to concatenation of two feature vectors
            nn.ReLU(), # activation function
            nn.Dropout(0.5), # increased dropout to prevent overfitting
            nn.Linear(256, 128), # hidden layer
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), # output layer
            nn.Tanh() # output between -1 and 1 for angle difference (normalized to [-1, 1])
        )

    def forward_once(self, x):
        x = self.backbone(x) # first pass through the ResNet backbone
        x = x.view(x.size(0), -1) # flatten the output tensor
        return x

    def forward(self, img1, img2):
        feat1 = self.forward_once(img1) # feature vector from the first image (512 dimensions)
        feat2 = self.forward_once(img2) # feature vector from the second image (512 dimensions)
        combined = torch.cat([feat1, feat2], dim=1) # concatenation of the two vectors (512 *2 = 1024 dimensions)

        similarity = self.fc_similarity(combined) # similarity score logits (pre-sigmoid)
        angle_diff = self.fc_angle(combined) # normalized angle difference in [-1, 1]

        return similarity, angle_diff

    def train_model(self, train_json='SiameseCNN/json_data/train_pairs.json', val_json='SiameseCNN/json_data/val_pairs.json', epochs=10, batch_size=16, lr=1e-4):
        """
        Train the Siamese CNN model.
        Args:
            train_json (str): Path to the training pairs JSON file.
            val_json (str): Path to the validation pairs JSON file.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            lr (float): Learning rate for the optimizer.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        model = self.to(device) # move the model to GPU if available

        train_dataset = TLESSDataset(train_json, crop_objects=True, augment=True)  # the augmentation is turned on for training
        val_dataset = TLESSDataset(val_json, crop_objects=True, augment=False)  # its turned off for validating

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4,pin_memory=True, persistent_workers=True, prefetch_factor=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True, persistent_workers=True, prefetch_factor=2) 
        
        pos_count = train_dataset.get_pos_count() # number of positive pairs(match)
        neg_count = train_dataset.get_neg_count() # number of negative pairs (no match)

        pos_weight = torch.tensor([neg_count / pos_count]).to(device)

        criterion_similarity = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # the pos_weight helps to balance the loss for imbalanced datasets
        criterion_angle = nn.SmoothL1Loss() # Huber loss for angle difference regression
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4) # Adam optimizer with weight decay for regularization

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3) # reduce the learning rate by 0.5 if validation loss doesnt improve for 3 epochs

        os.makedirs('checkpoints', exist_ok=True)
        best_val_loss = float('inf')
        patience = 7 # early stopping patience
        patience_counter = 0 # counter for early stopping

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

            train_loader_tqdm = tqdm(train_loader, desc="Training", leave=False) # progress bar for training
            for ref_img, query_img, match_label, angle_diff in train_loader_tqdm:
                ref_img = ref_img.to(device) # move tensors to GPU if available
                query_img = query_img.to(device) # move tensors to GPU if available
                match_label = match_label.to(device).unsqueeze(1) # reshape to (batch_size, 1)
                angle_diff = angle_diff.to(device).unsqueeze(1) / 180.0 # normalize angle difference to [0, 1]

                optimizer.zero_grad() # make the gradients zero before backpropagation

                similarity, pred_angle = model(ref_img, query_img) # forward pass

                loss_sim = criterion_similarity(similarity, match_label) # similarity loss
                loss_angle = criterion_angle(pred_angle, angle_diff) # angle difference loss
                loss = 0.2 * loss_sim + 10.0 * loss_angle  # combined loss with weights to balance the two tasks

                loss.backward() # backpropagation
                optimizer.step() # update the weights

                train_loss += loss.item() # total training loss
                train_loss_sim_total += loss_sim.item() # total similarity loss
                train_loss_angle_total += loss_angle.item() # total angle loss

                preds = (torch.sigmoid(similarity) > 0.5).float() # predicted labels based on sigmoid output
                train_correct += (preds == match_label).sum().item() # count correct predictions
                train_total += match_label.size(0) # total samples

                angle_error = torch.abs(pred_angle - angle_diff) * 180.0 # denormalize angle error to degrees
                train_angle_errors.extend(angle_error.detach().cpu().numpy()) # collect angle errors 

                train_loader_tqdm.set_postfix(loss=loss.item()) # update progress bar with current loss

            model.eval() # switch to evaluation mode
            val_loss = 0.0
            val_loss_sim_total = 0.0
            val_loss_angle_total = 0.0
            val_correct = 0
            val_total = 0
            val_angle_errors = []

            val_loader_tqdm = tqdm(val_loader, desc="Validation", leave=False) # progress bar for validation
            with torch.no_grad():
                for ref_img, query_img, match_label, angle_diff in val_loader_tqdm:
                    ref_img = ref_img.to(device) # move tensors to GPU if available
                    query_img = query_img.to(device) # move tensors to GPU if available
                    match_label = match_label.to(device).unsqueeze(1) # reshape to (batch_size, 1)
                    angle_diff = angle_diff.to(device).unsqueeze(1) / 180.0 # normalize angle difference to [0, 1]

                    similarity, pred_angle = model(ref_img, query_img) # forward pass

                    loss_sim = criterion_similarity(similarity, match_label) # similarity loss
                    loss_angle = criterion_angle(pred_angle, angle_diff) # angle difference loss
                    loss = 0.2 * loss_sim + 10.0 * loss_angle  # combined loss with weights to balance the two tasks

                    val_loss += loss.item() # total validation loss
                    val_loss_sim_total += loss_sim.item() # total similarity loss
                    val_loss_angle_total += loss_angle.item() # total angle loss

                    preds = (torch.sigmoid(similarity) > 0.5).float() # predicted labels based on sigmoid output
                    val_correct += (preds == match_label).sum().item() # count correct predictions
                    val_total += match_label.size(0) # total samples

                    angle_error = torch.abs(pred_angle - angle_diff) * 180.0 # denormalize angle error to degrees
                    val_angle_errors.extend(angle_error.cpu().numpy()) # collect angle errors

                    val_loader_tqdm.set_postfix(loss=loss.item()) # update progress bar with current loss

            # metrics calculations
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            train_angle_mae = sum(train_angle_errors) / len(train_angle_errors)
            val_angle_mae = sum(val_angle_errors) / len(val_angle_errors)

            # metrics logging for plotting later
            metrics_history['train_loss'].append(avg_train_loss)
            metrics_history['train_loss_sim'].append(train_loss_sim_total / len(train_loader))
            metrics_history['train_loss_angle'].append(train_loss_angle_total / len(train_loader))
            metrics_history['train_accuracy'].append(train_acc)
            metrics_history['train_angle_mae'].append(float(train_angle_mae))
            metrics_history['val_loss'].append(avg_val_loss)
            metrics_history['val_loss_sim'].append(val_loss_sim_total / len(val_loader))
            metrics_history['val_loss_angle'].append(val_loss_angle_total / len(val_loader))
            metrics_history['val_accuracy'].append(val_acc)
            metrics_history['val_angle_mae'].append(float(val_angle_mae))
            metrics_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            print(f"Epoch {epoch + 1} complete - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"  Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
            print(f"  Train Angle MAE: {float(train_angle_mae):.2f}°, Val Angle MAE: {float(val_angle_mae):.2f}°")

            scheduler.step(avg_val_loss) # step the scheduler based on validation loss

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss # if the validation loss improved, save the model
                patience_counter = 0
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch
                }, 'SiameseCNN/checkpoints/best_checkpoint.pth')
                print(f"Best model saved (Val Loss: {best_val_loss:.4f})")
            else: # else, increase the patience counter
                patience_counter += 1 
                if patience_counter >= patience: # if patience exceeded, stop training
                    break

        torch.save(model.state_dict(), 'SiameseCNN/checkpoints/final_model.pth') # save the final model

        with open('SiameseCNN/checkpoints/training_metrics.json', 'w') as f:
            json.dump(metrics_history, f, indent=2) # save training metrics to a json file for plotting
            f.flush()

        return model

    def load_model(self, checkpoint_path):
        """
        Load the model from a checkpoint
        Args:
            checkpoint_path (str): Path to the checkpoint file.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.load_state_dict(checkpoint)

        self.to(device) 
        self.eval() # set the model to evaluation mode
        return self, device

    def evaluate_model(self, checkpoint_path='SiameseCNN/checkpoints/best_checkpoint.pth', test_json='SiameseCNN/json_data/test_pairs.json', batch_size=16):
        """
        Evaluate the model on the test dataset.
        Args:
            checkpoint_path (str): Path to the model checkpoint (saved weights).
            test_json (str): Path to the json file with test pairs.
            batch_size (int): Batch size for evaluation.
        """
        model, device = self.load_model(checkpoint_path)

        test_dataset = TLESSDataset(test_json, crop_objects=True, augment=False) # no data augmentation for testing

        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4,pin_memory=True, persistent_workers=True, prefetch_factor=2) # data loader for test dataset
        
        test_correct = 0
        test_total = 0
        test_angle_errors = []

        output_trues = [] # true labels for the ROC curve
        output_preds = [] # predicted similarity scores for the ROC curve


        model.eval() # switch to evaluation mode
        test_loader_tqdm = tqdm(test_loader, desc="Testing", leave=False) # progress bar for testing
        with torch.no_grad(): # no gradient calculation needed for evaluation
            for ref_img, query_img, match_label, angle_diff in test_loader_tqdm:
                # Code below has an identical structure to the validation loop in the train_model method
                ref_img = ref_img.to(device)
                query_img = query_img.to(device)
                match_label = match_label.to(device).unsqueeze(1)
                angle_diff = angle_diff.to(device).unsqueeze(1) / 180.0

                similarity, pred_angle = model(ref_img, query_img)

                preds = (torch.sigmoid(similarity) > 0.5).float()
                test_correct += (preds == match_label).sum().item()
                test_total += match_label.size(0)

                output_trues.extend(match_label.cpu().numpy()) # add true labels for the ROC curve
                output_preds.extend(torch.sigmoid(similarity).cpu().numpy()) # add predicted similarity scores for the ROC curve

                angle_error = torch.abs(pred_angle - angle_diff) * 180.0 # denormalize angle error to degrees
                test_angle_errors.extend(angle_error.cpu().numpy()) # collect angle errors

        test_acc = 100.0 * test_correct / test_total
        test_angle_mae = sum(test_angle_errors) / len(test_angle_errors)

        print(f"Test Accuracy: {test_acc:.2f}%")
        print(f"Test Angle MAE: {float(test_angle_mae):.2f}°")

        self.make_plot(metrics_json='SiameseCNN/training_metrics/training_metrics.json', output_trues=output_trues, output_preds=output_preds) # plot training metrics and the ROC curve

    def make_plot(self, metrics_json='SiameseCNN/training_metrics/training_metrics.json', output_trues=None, output_preds=None):
        """
        Plot training metrics and the ROC curve.
        Args:
            metrics_json (str): Path to the JSON file containing training metrics.
            output_trues (list): True labels for the ROC curve.
            output_preds (list): Predicted similarity scores for the ROC curve.
        """
        with open(metrics_json, 'r') as f:
            metrics = json.load(f)

        epochs = range(1, len(metrics['train_loss']) + 1)

        fpr, tpr, _ = roc_curve(output_trues, output_preds)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 2, 1)
        plt.plot(epochs, metrics['train_loss'], label='Train Loss')
        plt.plot(epochs, metrics['val_loss'], label='Val Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 2)
        plt.plot(epochs, metrics['train_accuracy'], label='Train Accuracy')
        plt.plot(epochs, metrics['val_accuracy'], label='Val Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.title('Training and Validation Accuracy')
        plt.legend()
        plt.grid()
        
        plt.subplot(2, 2, 3)
        plt.plot(epochs, metrics['train_angle_mae'], label='Train Angle MAE')
        plt.plot(epochs, metrics['val_angle_mae'], label='Val Angle MAE')
        plt.xlabel('Epochs')
        plt.ylabel('Angle MAE (degrees)')
        plt.title('Training and Validation Angle MAE')
        plt.legend()
        plt.grid()

        plt.subplot(2, 2, 4)
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.grid()

        plt.tight_layout()
        plt.show()