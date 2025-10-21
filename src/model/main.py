from .model import SiameseCNN

class Main:
    def __init__(self):
        self.siameseCNN = SiameseCNN(pretrained=True, freeze_backbone=False)

    def run(self, train_json='train_pairs.json', val_json='val_pairs.json', epochs=10, batch_size=16, lr=1e-4):
        self.siameseCNN.train_model(train_json=train_json, val_json=val_json, epochs=epochs, batch_size=batch_size, lr=lr)

if __name__ == "__main__":
    main = Main()
    main.run()
    