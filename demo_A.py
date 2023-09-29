import torch
import torch.nn as nn
import timm
from torchvision import transforms as Transforms
import argparse
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='demo A')
    parser.add_argument('--model-weight', type=str, default='./swin_base_patch4_window7_224.pth',
                        help='the path of model weight')
    parser.add_argument('--image1', type=str, default='./image1.jpg', help='the path of image 1')
    parser.add_argument('--image2', type=str, default='./image2.jpg', help='the path of image 2')
    args = parser.parse_args()

    data_processor = Data_Processor(height=224, width=224)
    model = SwinTransformer(num_features=512).cuda()
    model.eval()

    weight_path = args.model_weight
    weight = torch.load(weight_path)
    model.load_state_dict(weight['state_dict'], strict=True)

    image1 = args.image1
    image2 = args.image2

    image1 = data_processor(Image.open(image1).convert('RGB')).cuda()
    image2 = data_processor(Image.open(image2).convert('RGB')).cuda()

    with torch.no_grad():
        A_feat = F.normalize(model(image1), dim=1).cpu()
        B_feat = F.normalize(model(image2), dim=1).cpu()
    simlarity = A_feat.matmul(B_feat.transpose(1, 0))
    print("\033[1;31m The similarity is {}\033[".format(simlarity[0, 0]))
