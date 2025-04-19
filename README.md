# da6401_assignment2-partB
#  Fine-Tuning ResNet18 on iNaturalist

This project fine-tunes a pretrained `ResNet18` model on the iNaturalist 12K dataset using PyTorch Lightning. It compares different fine-tuning strategies and includes training from scratch as a baseline.

---

##  Dataset

- **iNaturalist 12K** (10-class subset)
- Folder structure: `train/class_name`, `val/class_name`
- Preprocessing: Resize (256x256), Normalize, optional `RandomHorizontalFlip`

---

##  Fine-Tuning Strategies

| Strategy                    | Description                                         |
|----------------------------|-----------------------------------------------------|
| Freeze Final Layer         | Train only the last layer (`fc`)                   |
| Partial Unfreeze           | Unfreeze last few layers (e.g., `layer4`, `layer3`) |
| Gradual Unfreeze           | Unfreeze deeper layers one-by-one across epochs     |
| From Scratch               | No pretraining, train all layers                    |

All strategies are based on `torchvision.models.resnet18`.

---

## ⚙Training Pipeline

- Uses **PyTorch Lightning**
- Optimizer: SGD with momentum
- Logging: **Weights & Biases (WandB)**
- Early stopping + model checkpointing

### Example Run

```python
finetuned_model = train_and_finetune(
    epochs=10,
    batch_size=32,
    data_aug='y',
    learning_rate=0.001
)

