# check_swa.py
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from dataset import TamperedDataset
from train import build_resnet50_9ch, evaluate_with_tta, FocalLoss
from torch.optim.swa_utils import AveragedModel

device = torch.device("cuda")
normalize_9 = transforms.Normalize(
    mean=[0.485,0.456,0.406, 0.5,0.5,0.5, 0.5,0.5,0.5],
    std =[0.229,0.224,0.225, 0.5,0.5,0.5, 0.5,0.5,0.5],
)
val_dataset = TamperedDataset("val", image_size=(224,224),
                               normalize_transform=normalize_9)
val_loader  = DataLoader(val_dataset, batch_size=32, shuffle=False,
                          num_workers=4, pin_memory=True)

base_model = build_resnet50_9ch(5, pretrained=False).to(device)
swa_model  = AveragedModel(base_model)
swa_model.load_state_dict(torch.load("outputs0/swa_model.pth"))
swa_model.to(device)

criterion = FocalLoss().to(device)
acc, loss = evaluate_with_tta(swa_model, val_loader, device, criterion, n_tta=4)
print(f"SWA Val Acc (TTA x4): {acc:.4f}")
print(f"Best model was      : 0.9592")