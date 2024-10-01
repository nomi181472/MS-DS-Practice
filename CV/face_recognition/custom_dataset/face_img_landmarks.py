import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import pandas as pd
import PIL
import os
class FaceImageLandmarksDataset(Dataset):
    def __init__(self, root_dir, landmarks_extension, transform=None):
        self.root_dir: Path = Path(root_dir)
        self.transform = transform
        self.landmarks_extension = landmarks_extension
        
        
        data = self.set_data_paths(root_dir, ["*.jpg", "*.png"])
        
        
        self.df = self.make_csv(data)
        self.max_landmarks = self.get_max_landmarks()
        self.num_of_classes=self.df['id'].nunique()
        print(f"num of classes:{self.num_of_classes}")
    def get_max_landmarks(self):
        max_landmarks = 0
        for idx in range(len(self.df)):
            landmarks_path = self.df.iloc[idx]["landmark_path"]
            landmarks = np.load(landmarks_path)
            max_landmarks = max(max_landmarks, len(landmarks))
        return max_landmarks
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
            flag=False
            for path in paths:
                landmarks_path = self.get_landmark_path(path)
                if landmarks_path.exists() and path.exists():
                    records.append({
                        "image_path": str(path),
                        "landmark_path": str(landmarks_path),
                        "name": name,
                        "id":name_count
                    })
                    flag=True
            if flag:
                name_count += 1

        df = pd.DataFrame(records)
       
        return df

    def __len__(self):
        return len(self.df)
    def pad_landmarks(self, landmarks):
        padded_landmarks = np.zeros((self.max_landmarks, landmarks.shape[1]))
        padded_landmarks[:landmarks.shape[0], :] = landmarks
        return padded_landmarks
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self.df.iloc[idx]["image_path"]
        landmarks_path = self.df.iloc[idx]["landmark_path"]
        id = self.df.iloc[idx]["id"]

        image = PIL.Image.open(img_path)
        image = np.array(image)
        assert image.size != 0, f"[ERROR] The image {img_path} is empty or cannot be read."
        
        landmarks = np.load(landmarks_path)
        #landmarks = self.pad_landmarks(landmarks)
        sample = {'id': id, 'landmarks': torch.tensor(landmarks, dtype=torch.float32),'image': torch.tensor(image, dtype=torch.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
    