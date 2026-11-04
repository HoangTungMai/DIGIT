"""U-Net alternative to TouchNet for DIGIT surface gradients.

Takes the same 5-channel input as py3DCal's TouchNet (BGR difference to the reference / 255,
then 1-based x and y pixel coordinates) and returns the same 2 gradient channels, so it is a
drop-in replacement trained by train_unet.py. Coordinates are rescaled to (0, 1] internally.

TouchNet keeps all 9 convolutions at full resolution (~298 GMAC at 320x240); this U-Net
downsamples 4 times (to 20x15) and needs ~14 GMAC at base=32. Frame sides must be
divisible by 2**levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def double_conv(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class UNet(nn.Module):
    def __init__(self, base=32, levels=4):
        super().__init__()
        self.base, self.levels = base, levels
        ch = [base * 2 ** i for i in range(levels + 1)]
        self.down = nn.ModuleList([double_conv(5, ch[0])]
                                  + [double_conv(ch[i], ch[i + 1]) for i in range(levels)])
        self.up = nn.ModuleList([nn.ConvTranspose2d(ch[i + 1], ch[i], 2, stride=2)
                                 for i in reversed(range(levels))])
        self.decode = nn.ModuleList([double_conv(2 * ch[i], ch[i]) for i in reversed(range(levels))])
        self.head = nn.Conv2d(ch[0], 2, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        if h % 2 ** self.levels or w % 2 ** self.levels:
            raise ValueError(f"Frame sides must be divisible by {2 ** self.levels}, got {h}x{w}")
        x = torch.cat([x[:, :3], x[:, 3:4] / w, x[:, 4:5] / h], dim=1)
        skips = []
        for i, block in enumerate(self.down):
            x = block(x if i == 0 else F.max_pool2d(x, 2))
            skips.append(x)
        skips.pop()
        for up, block in zip(self.up, self.decode):
            x = block(torch.cat([up(x), skips.pop()], dim=1))
        return self.head(x)
