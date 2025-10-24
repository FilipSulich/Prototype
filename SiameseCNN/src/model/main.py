from .model import SiameseCNN
from pathlib import Path

class Main:
    def __init__(self):
        """
        Main class to handle training and evaluation of the model.
        """
        self.siameseCNN = SiameseCNN(freeze_backbone=False) # initialize the Siamese CNN model

    def train(self, train_json='SiameseCNN/json_data/train_pairs.json', val_json='SiameseCNN/json_data/val_pairs.json', epochs=10, batch_size=16, lr=1e-4):
        self.siameseCNN.train_model(train_json=train_json, val_json=val_json, epochs=epochs, batch_size=batch_size, lr=lr) # train the model

    def evaluate(self, checkpoint_path='SiameseCNN/checkpoints/final_model.pth', test_json='SiameseCNN/json_data/test_pairs.json', batch_size=16):
        self.siameseCNN.evaluate_model(checkpoint_path=checkpoint_path, test_json=test_json, batch_size=batch_size) # evaluate the model

if __name__ == "__main__":
    main = Main()
    # main.train()    
    main.evaluate()