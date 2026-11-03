"""TouchNet with its heaviest layers (conv4-conv6, ~84% of compute) run at 1/scale resolution.

Same weights layout as py3DCal's TouchNet, so it starts from the pretrained DIGIT weights
and is fine-tuned by train_fast_touchnet.py.
"""

import torch
import torch.nn.functional as F


class FastTouchNet(torch.nn.Module):
    def __init__(self, net, scale=2):
        super().__init__()
        self.net = net
        self.scale = scale

    def forward(self, x):
        n = self.net
        x = F.relu(n.bn1(n.conv1(x)))
        x = F.relu(n.bn2(n.conv2(x)))
        x = F.relu(n.bn3(n.conv3(x)))
        size = x.shape[-2:]
        if self.scale > 1:
            x = F.avg_pool2d(x, self.scale)
        x = F.relu(n.bn4(n.conv4(x)))
        x = F.relu(n.bn5(n.conv5(x)))
        x = F.relu(n.bn6(n.conv6(x)))
        if self.scale > 1:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        x = F.relu(n.bn7(n.conv7(x)))
        x = F.relu(n.bn8(n.conv8(x)))
        return n.conv9(x)
