#from custom_dataset.face_img_landmarks import FaceImageLandmarksDataset

import torch
import torchvision.transforms.functional as TF
import numpy as np
import os
import PIL
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class FaceImageLandmarksDataset(Dataset):
    def __init__(self, root_dir, landmarks_extension, transform=None):
        self.root_dir: Path = Path(root_dir)
        self.transform = transform
        self.landmarks_extension = landmarks_extension
        
        data = self.set_data_paths(root_dir, ["*.jpg", "*.png"])
        
        self.df = self.make_csv(data)
        self.max_landmarks = self.get_max_landmarks()
        self.max_image_size = self.get_max_image_size()
        self.num_of_classes = self.df['id'].nunique()
        print(f"num of classes: {self.num_of_classes}")
    
    def get_max_landmarks(self):
        max_landmarks = 0
        for idx in range(len(self.df)):
            landmarks_path = self.df.iloc[idx]["landmark_path"]
            landmarks = np.load(landmarks_path)
            max_landmarks = max(max_landmarks, len(landmarks))
        return max_landmarks

    def get_max_image_size(self):
        # Calculate the maximum width and height across the dataset
        max_width, max_height = 0, 0
        for idx in range(len(self.df)):
            img_path = self.df.iloc[idx]["image_path"]
            with PIL.Image.open(img_path) as img:
                width, height = img.size
                max_width = max(max_width, width)
                max_height = max(max_height, height)
        return max_width, max_height

    def set_data_paths(self, root_dir, extensions: list = list) -> dict:
        all_names = os.listdir(root_dir)
        data_paths = {}
        for name in all_names:
            new_path = self.root_dir.joinpath(name)
            paths = []
            for ext in extensions:
                paths += list(new_path.rglob(ext))
            data_paths[name] = paths
        return data_paths

    def get_landmark_path(self, path):
        landmarks_path = Path(str(path)[:-4] + self.landmarks_extension)
        return landmarks_path

    def make_csv(self, data: dict):
        records = []
        name_count = 0
        for name, paths in data.items():
            flag = False
            for path in paths:
                landmarks_path = self.get_landmark_path(path)
                if landmarks_path.exists() and path.exists():
                    records.append({
                        "image_path": str(path),
                        "landmark_path": str(landmarks_path),
                        "name": name,
                        "id": name_count
                    })
                    flag = True
            if flag:
                name_count += 1

        df = pd.DataFrame(records)
        return df

    def __len__(self):
        return len(self.df)

    def pad_image(self, image):
        # Get the current image size
        width, height = image.size
        
        # Calculate padding needed to reach max size
        max_width, max_height = self.max_image_size
        pad_width = max_width - width
        pad_height = max_height - height
        
        # Add padding equally on both sides
        padding = (pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2)
        
        # Apply padding
        padded_image = TF.pad(image, padding, fill=0)
        return padded_image

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]["image_path"]
        landmarks_path = self.df.iloc[idx]["landmark_path"]
        id = self.df.iloc[idx]["id"]

        image = PIL.Image.open(img_path).convert("RGB")
        # Pad the image to match the largest size in the dataset
        image = self.pad_image(image)
        image = np.array(image)
        
        landmarks = np.load(landmarks_path)
        sample = {'id': id, 'landmarks': torch.tensor(landmarks, dtype=torch.float32), 
                  'image': torch.tensor(image, dtype=torch.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample

    
