import json
import sys
import matplotlib.pyplot as plt, numpy as np, os, torch, random
import cv2
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torchvision import transforms, models
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class LegosDataset(Dataset):
    def __init__(self, images_filenames, transform=None):
        self.images_filenames = images_filenames
        self.transform = transform

    def __len__(self):
        return len(self.images_filenames)

    def __getitem__(self, idx):
        image_filename = self.images_filenames[idx]

        # Read image
        image = cv2.imread(image_filename)

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply the same data augmentation to both input image and target mask
        if self.transform is not None:
            image = self.transform(image)

        return image

def input(inputs):
    with open(inputs, 'r') as file:
        data = json.load(file)

    list = []
    for id, filename in enumerate(data["image_files"]):
        list.append(filename)

    return list

def output(results, output_file):
    with open(output_file, 'w') as file:
        json.dump({"results": results}, file, indent=4)

def dataLoader(image_paths):
    image_paths = np.array(image_paths)

    batch_size = 1
    num_workers = 2

    # Define transformations to be applied to data
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    dataset = LegosDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=False)

    return dataloader

def epoch_iter(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_predictions = []

    for images in tqdm(dataloader):
        images = images.to(device)

        with torch.set_grad_enabled(False):
            outputs = model(images)

        probs = F.softmax(outputs, dim=1)

        # print("probs: ", probs)

        all_predictions.extend(outputs.argmax(dim=1).detach().cpu().numpy())

    return all_predictions

def main(input_file, output_file):
    # Load images
    image_paths = input(input_file)
    
    dataloader = dataLoader(image_paths)

    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 32)
    model = model.to(device)

    model_name = "LEGO_resnet18"

    checkpoint = torch.load(model_name + '_best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model'])

    predictions = epoch_iter(model, dataloader)

    # Output results
    results = [{"file_name" : image_paths[i], "num_detections" : int(predictions[i] + 1)} for i in range(len(image_paths))]
    output(results, output_file)



if __name__ == '__main__':
    if len(sys.argv) == 3:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        main(input_file, output_file)
    else:
        print("Usage: python lego.py <input_file> <output_file>")
        sys.exit(1)
