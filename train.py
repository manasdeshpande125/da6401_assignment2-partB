# Imports
import os
os.system("pip install pytorch_lightning")
import torch
import os
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
import torchvision
from pytorch_lightning.loggers import WandbLogger
import argparse  # Missing import for argparse

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Optional: Enable DataParallel if multiple GPUs
def prepare_model_for_device(model):
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    return model.to(device)

# Dataset setup
if not os.path.exists("inaturalist_12K"):
    os.system("wget https://storage.googleapis.com/wandb_datasets/nature_12K.zip -O nature_12K.zip")
    os.system("unzip -q nature_12K.zip")
    os.system("rm nature_12K.zip")


# WandB logger setup (default)
wandb_logger = WandbLogger(project='DA6401-Assignment-2')

# ---------- DATA LOADER ---------- #
def load_data(batch_size=32, data_aug='n'):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip() if data_aug == 'y' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    train_dataset = datasets.ImageFolder(root='inaturalist_12K/train', transform=transform)
    val_dataset = datasets.ImageFolder(root='inaturalist_12K/val', transform=transform)

    val_size = int(0.2 * len(train_dataset))
    train_size = len(train_dataset) - val_size
    train_ds, val_ds = random_split(train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

# ---------- MODEL CLASSES ---------- #
class FineTuneResNet18(pl.LightningModule):
    def __init__(self, learning_rate=0.001, freeze_layers=True, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(pretrained=True)
        if freeze_layers:
            for param in self.model.parameters():
                param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x): return self.model(x)

    def configure_optimizers(self):
        return optim.SGD(self.model.parameters(), lr=self.hparams.learning_rate, momentum=0.9)

    def shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self.shared_step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self.shared_step(batch, 'val')
    def test_step(self, batch, batch_idx): return self.shared_step(batch, 'test')


class PartialFineTuneResNet18(FineTuneResNet18):
    def __init__(self, learning_rate=0.001, unfreeze_from_layer=6, num_classes=10):
        super().__init__(learning_rate=learning_rate, freeze_layers=True, num_classes=num_classes)
        ct = 0
        for child in self.model.children():
            if ct >= unfreeze_from_layer:
                for param in child.parameters():
                    param.requires_grad = True
            ct += 1


class GradualUnfreezeResNet18(pl.LightningModule):
    def __init__(self, learning_rate=0.001, num_classes=10):
        super().__init__()
        self.save_hyperparameters()
        self.model = torchvision.models.resnet18(pretrained=True)

        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.fc.parameters():
            param.requires_grad = True

        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.unfreeze_schedule = [self.model.layer4, self.model.layer3, self.model.layer2]
        self.unfreeze_index = 0

    def forward(self, x): return self.model(x)

    def configure_optimizers(self):
        return optim.SGD([
            {"params": self.model.fc.parameters(), "lr": self.hparams.learning_rate},
            {"params": self.model.layer4.parameters(), "lr": self.hparams.learning_rate * 0.5},
            {"params": self.model.layer3.parameters(), "lr": self.hparams.learning_rate * 0.1},
            {"params": self.model.layer2.parameters(), "lr": self.hparams.learning_rate * 0.05},
        ], momentum=0.9)

    def on_train_epoch_start(self):
        if self.unfreeze_index < len(self.unfreeze_schedule):
            for param in self.unfreeze_schedule[self.unfreeze_index].parameters():
                param.requires_grad = True
            print(f"Unfroze layer: {self.unfreeze_index}")
            self.unfreeze_index += 1

    def shared_step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = nn.CrossEntropyLoss()(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f'{stage}_loss', loss, prog_bar=True, on_epoch=True)
        self.log(f'{stage}_acc', acc, prog_bar=True, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx): return self.shared_step(batch, 'train')
    def validation_step(self, batch, batch_idx): return self.shared_step(batch, 'val')
    def test_step(self, batch, batch_idx): return self.shared_step(batch, 'test')


class TrainResNet18FromScratch(FineTuneResNet18):
    def __init__(self, learning_rate=0.001, num_classes=10):
        super().__init__(learning_rate=learning_rate, freeze_layers=False, num_classes=num_classes)
        self.model = torchvision.models.resnet18(pretrained=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

# ---------- TRAINING WRAPPER ---------- #
def train_and_finetune(model_type='full_finetune', epochs=10, batch_size=32, data_aug='y', learning_rate=0.001):
    train_loader, val_loader = load_data(batch_size, data_aug)

    # Logger update
    wandb_logger.experiment.name = model_type

    # Select model
    if model_type == 'full_finetune':
        model = FineTuneResNet18(learning_rate=learning_rate)
    elif model_type == 'partial_finetune':
        model = PartialFineTuneResNet18(learning_rate=learning_rate, unfreeze_from_layer=6)
    elif model_type == 'gradual_unfreeze':
        model = GradualUnfreezeResNet18(learning_rate=learning_rate)
    elif model_type == 'from_scratch':
        model = TrainResNet18FromScratch(learning_rate=learning_rate)
    else:
        raise ValueError("Invalid model_type. Choose from: 'full_finetune', 'partial_finetune', 'gradual_unfreeze', 'from_scratch'.")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=f"{model_type}" + "-{epoch:02d}-{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_weights_only=True
    )

    early_stop_callback = EarlyStopping(monitor='val_loss', patience=3, mode='min')

    trainer = Trainer(
        max_epochs=epochs,
        callbacks=[checkpoint_callback, early_stop_callback],
        logger=wandb_logger,
        accelerator='auto',
        devices=1 if torch.cuda.is_available() else None,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, val_loader)

    return model



# Main entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN on iNaturalist_12K with PyTorch Lightning")

    # Default values chosen from the best-performing model
    parser.add_argument("--wandb_entity", "-we",help = "Wandb Entity used to track experiments in the Weights & Biases dashboard.", default="cs24m024")
    parser.add_argument("--wandb_project", "-wp",help="Project name used to track experiments in Weights & Biases dashboard", default="Trial")
    parser.add_argument("--model_type", "-mt",help="Strategy used for fine tuning", default="gradual_unfreeze")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_count", type=int, default=64)
    parser.add_argument("--data_aug", type=str, choices=['y', 'n'], default='y')
    parser.add_argument("--lr_rate", type=float, default=0.001)

    args = parser.parse_args()
    # print(args.epochs)
    import wandb
    wandb.login()
    wandb.init(project=args.wandb_project, entity=args.wandb_entity)

    # Train using gradual unfreezing strategy
    finetuned_model = train_and_finetune(
        model_type=args.model_type,  #options="full_finetune","partial_finetune","gradual_unfreeze","from_scratch"
        epochs=args.epochs,
        batch_size=args.batch_count,
        data_aug=args.data_aug,
        learning_rate=args.lr_rate
    )
