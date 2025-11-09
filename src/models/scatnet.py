import torch.nn as nn
class SCATNet(nn.Module):
    def __init__(self, in_bands:int, num_classes:int, ch:int=16):
        super().__init__()
        self.stem = nn.Conv3d(1, ch, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(ch)
        self.act = nn.ReLU(inplace=True)
        self.mix = nn.Conv3d(ch, ch, 1)
        self.pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.head = nn.Linear(ch, num_classes)
    def forward(self, x):  # x: [B, Bands, H, W]
        x = x.unsqueeze(1)           # [B,1,B,H,W]
        x = self.act(self.bn(self.stem(x)))
        x = self.mix(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.head(x)
