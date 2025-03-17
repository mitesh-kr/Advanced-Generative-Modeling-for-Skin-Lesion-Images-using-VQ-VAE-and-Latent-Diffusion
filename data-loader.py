import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.img_folder = img_folder
        self.transform = transform
        self.image_names = os.listdir(img_folder)  # Get the list of image names in the folder

    def __getitem__(self, index):
        # Get image name from the list
        img_name = self.image_names[index]

        # Open image
        image = Image.open(os.path.join(self.img_folder, img_name))

        # Apply transformations if any
        if self.transform is not None:
            image = self.transform(image)

        return image

    def __len__(self):
        return len(self.image_names)

def get_data_loader(img_folder, batch_size, image_size=64):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in range(3)], [0.5 for _ in range(3)]),
    ])
    
    dataset = CustomDataset(img_folder=img_folder, transform=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    return data_loader
