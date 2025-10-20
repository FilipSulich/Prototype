import torch
import torch.nn as nn
import torchvision.models as models

class SiameseCNN(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True):
        super(SiameseCNN, self).__init__()

        resnet = models.resnet18(pretrained=pretrained) # we use ResNet18 as the backbone
        self.backbone = nn.Sequential(*list(resnet.children())[:-1]) # we use all the layers from ResNet18 except the final fully connected layer - we only need the feature extractor

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False # freeze backbone parameters (so we dont change the ResNet weights)

        self.fc_similarity = nn.Sequential(
            nn.Linear(512 * 2, 256), # input size is 512*2 due to concatenation of two feature vectors
            nn.ReLU(),
            nn.Dropout(0.3), 
            nn.Linear(256, 1), 
            nn.Sigmoid() # output between 0 and 1 for similarity score
        )

        self.fc_angle = nn.Sequential(
            nn.Linear(512 * 2, 256), # input size is 512*2 due to concatenation of two feature vectors
            nn.ReLU(), 
            nn.Dropout(0.3), # dropout to prevent overfitting
            nn.Linear(256, 1), 
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
