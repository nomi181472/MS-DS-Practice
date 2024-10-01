import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.utils.tensorboard import SummaryWriter

class LandmarkClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LandmarkClassifier, self).__init__()

        # 1D Convolution layers
        self.conv1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        #self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        #self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * int(input_size/2), 512)  # First fully connected layer after Conv1D
        self.bone = nn.ModuleList([nn.Linear(512, 512) for _ in range(4)])  # Bone layers
        self.fcsl = nn.Linear(512, 256)  # Second FC layer
        self.fcl = nn.Linear(256, num_classes)  # Output layer

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)  # Dropout to prevent overfitting
    def update_model_for_new_dataset(self,output_size,):
        self.fcl=nn.Linear(256, output_size)
    def forward(self, x):
        
        x = x.permute(0, 3, 2, 1)  
        x = x.squeeze(-1) 
           
        x = self.relu(self.conv1(x))
        #print(f"shape:{x.shape}")
        #x = self.relu(self.conv2(x))
        #x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        for fc in self.bone:
            x = self.relu(fc(x))
            x = self.dropout(x)
        x = self.dropout(x)
        
        x = self.relu(self.fcsl(x))
        
        logits = self.fcl(x)
        probs = F.softmax(logits, dim=1)  # Apply softmax to get probabilities

        return probs, logits


