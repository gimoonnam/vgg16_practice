
import os
from datetime import datetime
import torch
import torch.nn as nn 


class N_conv(nn.Module):
    """
    This a class for defining the N filters 
    Attributes
    ----------
    conv  : nn.Sequential
        defines the train model
    """
    def __init__(self, in_channels, out_channels, N=2):
        super().__init__()
        model = []
        model.append(nn.Conv2d(in_channels, out_channels, kernel_size=(3,3), padding=(1,1)))
        model.append(nn.ReLU(True))
        
        for i in range(N-1):
            model.append(nn.Conv2d(out_channels,out_channels,kernel_size=(3,3),padding=(1,1)))
            model.append(nn.ReLU(True))

        model.append(nn.MaxPool2d(kernel_size=(2,2),stride=(2,2)))
        self.conv = nn.Sequential(*model)

    def forward(self,x):
        return self.conv(x)
    

class VGG16(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 2, init_weights: bool = True):
        super().__init__()
        self.conv1 = N_conv(in_ch, 64)
        self.conv2 = N_conv(64, 128)
        self.conv3 = N_conv(128, 256, N=3)
        self.conv4 = N_conv(256, 512, N=3)
        self.conv5 = N_conv(512, 512, N=3)
        self.avgpool = nn.AdaptiveAvgPool2d((7,7))
        self.linear1 = nn.Linear(512*7*7,4096)
        self.linear2 = nn.Linear(4096,4096)
        self.relu = nn.ReLU(True)
        self.dropout = nn.Dropout(0.3)
        self.linear3 = nn.Linear(4096, out_ch)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x



def save_checkpoint(save_path: str, batch_size: int, epoch: int, model: VGG16, optimizer: torch.optim.Optimizer, loss: float):
    checkpoint = {
        'batch_size': batch_size,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'saved_datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    save_pth = os.path.join(save_path, 'checkpoint-' + checkpoint['saved_datetime'] + '.pth')
    torch.save(checkpoint, save_pth)


def load_checkpoint(checkpoint_path: str, model: VGG16, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch: int = checkpoint['epoch']
    loss: float = checkpoint['loss']

    return model, optimizer, epoch, loss
