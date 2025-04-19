# da6401_assignment2-partB
#  Fine-Tuning ResNet18 on iNaturalist

This project fine-tunes a pretrained `ResNet18` model on the iNaturalist 12K dataset using PyTorch Lightning. It compares different fine-tuning strategies and includes training from scratch as a baseline.

---

##  Dataset

- **iNaturalist 12K** (10-class subset)
- Folder structure: `train/class_name`, `val/class_name`
- Preprocessing: Resize (224x2224), Normalize

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


## Training Pipeline

- Uses **PyTorch Lightning**
- Optimizer: SGD with momentum
- Logging: **Weights & Biases (WandB)**
- Early stopping + model checkpointing

---


## Highlights

-  **Transfer Learning Works!**
  - Using a model pretrained on ImageNet significantly improves convergence and accuracy on small datasets like iNaturalist.
  
- **Fine-tuning Strategy Matters**
  - Freezing all layers except the final FC layer is fast but less accurate.
  - Partially unfreezing deeper layers (e.g., `layer4`, `layer3`) offers a great balance between speed and accuracy.
  - Gradual unfreezing across epochs helps avoid catastrophic forgetting and allows smoother adaptation to the target dataset.

- **Layer-wise Learning Control**
  - Gradual unfreezing + differential learning rates provide fine control over what the model learns and when.

- **Logging + Monitoring**
  - Integrated with Weights & Biases (WandB) for tracking metrics, losses, and training progress.
---

## Requirements

Install the required packages with pip:
torch>=1.12.0
torchvision>=0.13.0
pytorch-lightning>=2.0.0
wandb>=0.15.0

### Wandb Link
https://api.wandb.ai/links/cs24m024-iit-madras/cboyyrhf

### Way to run train.py
!python train.py --wandb_entity "cs24m024-iit-madras" --wandb_project "da6401-assignment-2" --epochs 3 --batch_count 32 --data_aug 'y' --lr_rate 0.0005 --model_type "from_scratch"

### Example Run

```python
finetuned_model = train_and_finetune(
    epochs=10,
    batch_size=32,
    data_aug='y',
    learning_rate=0.001
)

----









