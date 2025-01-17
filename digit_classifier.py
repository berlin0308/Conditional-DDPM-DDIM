import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import detectors
import timm
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image

MNISTM_MEAN = [0.5, 0.5, 0.5]
MNISTM_STD = [0.5, 0.5, 0.5]
SVHN_MEAN = [0.5, 0.5, 0.5]
SVHN_STD = [0.5, 0.5, 0.5]

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state['state_dict'])
    print(f'Model loaded from {checkpoint_path}')

class CustomDataset(Dataset):
    def __init__(self, path, transform):
        self.img_dir = path
        self.transform = transform
        self.data = []
        self.labels = []
        for filename in os.listdir(self.img_dir):
            self.data.append(os.path.join(self.img_dir, filename))
            self.labels.append(int(filename[0]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        return self.transform(img), self.labels[idx]

class MnistmClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def evaluate_model(model, data_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", help="path to the folder containing 'mnistm' and 'svhn' subfolders", type=str)
    parser.add_argument("--checkpoint", help="path to the MNIST-M model checkpoint", type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MNIST-M model setup
    mnistm_model = MnistmClassifier().to(device)
    load_checkpoint(args.checkpoint, mnistm_model)
    mnistm_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(MNISTM_MEAN, MNISTM_STD)
    ])
    mnistm_dataset = CustomDataset(os.path.join(args.folder, 'mnistm'), mnistm_transform)
    mnistm_loader = torch.utils.data.DataLoader(mnistm_dataset, batch_size=32, shuffle=False)

    # SVHN model setup (using timm)
    svhn_model = timm.create_model("resnet34_svhn", pretrained=True, num_classes=10).to(device)
    svhn_transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(SVHN_MEAN, SVHN_STD)
    ])
    svhn_dataset = CustomDataset(os.path.join(args.folder, 'svhn'), svhn_transform)
    svhn_loader = torch.utils.data.DataLoader(svhn_dataset, batch_size=32, shuffle=False)

    # Evaluate MNIST-M model
    mnistm_correct, mnistm_total = evaluate_model(mnistm_model, mnistm_loader, device)
    # print(f'MNIST-M acc = {mnistm_correct/mnistm_total:.4f} (correct/total = {mnistm_correct}/{mnistm_total})')

    # Evaluate SVHN model
    svhn_correct, svhn_total = evaluate_model(svhn_model, svhn_loader, device)
    print(f'MNIST-M acc = {mnistm_correct/mnistm_total:.4f} (correct/total = {mnistm_correct}/{mnistm_total})')
    print(f'SVHN acc = {svhn_correct/svhn_total:.4f} (correct/total = {svhn_correct}/{svhn_total})')
    print(f'acc = {((mnistm_correct/mnistm_total + svhn_correct/svhn_total) / 2):.4f}')