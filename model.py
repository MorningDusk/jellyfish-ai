import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ResNetYOLO(nn.Module):
    def __init__(self, num_classes=80, hyp=None):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove the final FC layer
        
        self.num_classes = num_classes
        self.detect = Detect(num_classes)
        
        # Calculate output dimensions
        self.no = num_classes + 5  # number of outputs per anchor (class + 5)
        self.na = 3  # number of anchors
        
        # Single detection head
        self.head = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, self.na * self.no, kernel_size=1)  # Updated output channels
        )
        
        # Initialize hyperparameters
        self.hyp = hyp if hyp is not None else {
            'anchor_t': 4.0,
            'gr': 1.0  # giou loss ratio
        }
        self.gr = self.hyp.get('gr', 1.0)

    @property
    def nl(self):
        return 1  # Single detection layer
    
    @property
    def anchors(self):
        return self.detect.anchors

    def forward(self, x):
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        print(f"Features shape after backbone: {features.shape}")

        features = self.head(features)
        print(f"Features shape after head: {features.shape}")

        return [features]  # Return as list for compatibility with YOLOv5 structure

class Detect(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.na = 3  # number of anchors
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = 1  # number of detection layers
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        
        # Define anchors
        self.register_buffer('anchors', torch.tensor([
            [10, 13], [16, 30], [33, 23]  # Single scale anchors
        ]).float())

    def forward(self, x):
        z = []
        for i in range(self.nl):
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                    self.anchor_grid[i] = self.anchors[i].clone().view(1, -1, 1, 1, 2).to(x[i].device)
                
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        
        return x[0] if self.training else torch.cat(z, 1)

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

def build_model(num_classes=80):
    return ResNetYOLO(num_classes=num_classes)