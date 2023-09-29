# ISR_ICCV2023_Oral
**[ICCV2023 Oral] Identity-Seeking Self-Supervised Representation Learning for Generalizable Person Re-identification.**

[[ArXiv](https://arxiv.org/pdf/2308.08887.pdf)] [[Demo](https://colab.research.google.com/drive/1MqEJ_O-e753N9NEVkvYcMZmlHlIWgcv6#scrollTo=hPiYsyp-hZbb)][![Demo In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MqEJ_O-e753N9NEVkvYcMZmlHlIWgcv6#scrollTo=hPiYsyp-hZbb)

ISR is a domain-generalizable person ReID model. It is trained with 47.8M person images extracted from 7.4K video clips in an unsupervised manner. ISR can not only be tested directly in unknown domains, but also can quickly adapt to new enviroments, showing good domain generalization and domain adaptation capabilities. Therefore, ISR has a more substantial potential for real-world applications.

## Update
2023-08-15: Update the trained model; The other code is coming soon.

## Trained model
Swin-Transformer---[swin_base_patch4_window7_224.pth](https://drive.google.com/file/d/1fB-5SaaUf3ZVnnSkQAVhBE3VAJ6u9e0w/view?usp=sharing)

## Demo
The demo has been uploaded to [colab](https://colab.research.google.com/drive/1MqEJ_O-e753N9NEVkvYcMZmlHlIWgcv6#scrollTo=-r1xlWpD0w5E)

## Evaluation
Download the trained weight, and then load it:
```python'''
import torch
import torch.nn as nn
import timm

import torch
import torch.nn as nn
import timm
from torchvision import transforms as Transforms
from PIL import Image
import torch.nn.functional as F
import os


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


class Data_Processor(object):
    def __init__(self, height, width):
        self.height = height
        self.width = width
        self.transformer = Transforms.Compose([
            Transforms.Resize((self.height, self.width)),
            Transforms.ToTensor(),
            Transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, img):
        return self.transformer(img).unsqueeze(0)

data_processor = Data_Processor(height=224, width=224)

ckpt = './swin_base_patch4_window7_224.pth'  # The path of the trained weight
ckpt = torch.load(ckpt_path)
model = SwinTransformer(num_features=512).cuda()
model.load_state_dict(ckpt['state_dict'], strict=True)
```

## Visualization demo
![Alt Text](https://github.com/dcp15/ISR_ICCV2023_Oral/blob/main/demo/demo-v1.gif)
