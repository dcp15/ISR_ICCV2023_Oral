# ISR_ICCV2023_Oral
[ICCV2023 Oral] Identity-Seeking Self-Supervised Representation Learning for Generalizable Person Re-identification. [arxiv](https://arxiv.org/pdf/2308.08887.pdf)

## Update
2023-08-15: Update the trained model; The other code is comming soon.

## Trained model
Swin-Transformer---[swin_base_patch4_window7_224.pth](https://drive.google.com/file/d/1Lb-oxhUSU38fNAucBBqIpRi2eQUjTXnP/view?usp=sharing)

## Evaluation
Download the trained weight, and then load it:
```python'''
import torch
import torch.nn as nn
import timm

class SwinTransformer(nn.Module):

    def __init__(self, num_features=512):
        super(SwinTransformer, self).__init__()
        self.model = timm.create_model('swin_base_patch4_window7_224')
        self.num_features = num_features
        self.feat = nn.Linear(1024, num_features) if num_features > 0 else None

    def forward(self, x):
        x = self.model.forward_features(x)
        if not self.feat is None:
            x = self.feat(x)
        return x

ckpt = './swin_base_patch4_window7_224.pth'  # The path of the trained weight
model = SwinTransformer(num_features=512)
ckpt = torch.load(ckpt_path)
model.load_state_dict(ckpt['model'])
```

## Visualization demo
![Alt Text](https://github.com/dcp15/ISR_ICCV2023_Oral/blob/main/demo/demo-v1.gif)
