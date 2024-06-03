from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class PhotoSketchDataset(Dataset):
    def __init__(self, root_sketch, root_photo, transform=None):
        self.root_sketch = root_sketch
        self.root_photo = root_photo
        self.transform = transform

        self.sketch_images = os.listdir(root_sketch)
        self.photo_images = os.listdir(root_photo)
        self.length_dataset = max(len(self.sketch_images), len(self.photo_images)) # 1000, 1500
        self.sketch_len = len(self.sketch_images)
        self.photo_len = len(self.photo_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        sketch_img = self.sketch_images[index % self.sketch_len]
        photo_img = self.photo_images[index % self.photo_len]

        sketch_path = os.path.join(self.root_sketch, sketch_img)
        photo_path = os.path.join(self.root_photo, photo_img)

        sketch_img = np.array(Image.open(sketch_path).convert("RGB"))
        photo_img = np.array(Image.open(photo_path).convert("RGB"))

        if self.transform:
            augmentations = self.transform(image=sketch_img, image0=photo_img)
            sketch_img = augmentations["image"]
            photo_img = augmentations["image0"]

        return sketch_img, photo_img
