import torch
from PIL import Image
import albumentations as A
# Import ToTensorV2 from albumentations.pytorch
from albumentations.pytorch import ToTensorV2

LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN = "gen.pth.tar"
CHECKPOINT_DISC = "disc.pth.tar"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 100000
BATCH_SIZE = 16
NUM_WORKERS = 4
HIGH_RES = 96
LOW_RES = HIGH_RES
IMG_CHANNELS = 3

highres_transform = A.Compose([
    # Fix the square brackets
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

lowres_transform = A.Compose([
    A.Resize(width=LOW_RES, height=LOW_RES, interpolation=Image.BICUBIC),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2(),
])

both_transforms = A.Compose([
    # Fix the typo in 'RandomCrop'
    A.RandomCrop(width=HIGH_RES, height=HIGH_RES),
    A.HorizontalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
])
