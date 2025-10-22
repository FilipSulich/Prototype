from .model import SiameseCNN
from pathlib import Path

class Main:
    def __init__(self):
        self.siameseCNN = SiameseCNN(freeze_backbone=False)

    def train(self, train_json='json_data/train_pairs.json', val_json='json_data/val_pairs.json', epochs=10, batch_size=16, lr=1e-4):
        self.siameseCNN.train_model(train_json=train_json, val_json=val_json, epochs=epochs, batch_size=batch_size, lr=lr)

    def evaluate(self, checkpoint_path='checkpoints/best_checkpoint.pth', test_json='json_data/test_pairs.json', batch_size=16):
        self.siameseCNN.evaluate_model(checkpoint_path=checkpoint_path, test_json=test_json, batch_size=batch_size)

if __name__ == "__main__":
    main = Main()
    # main.train()    
    
    main.evaluate()