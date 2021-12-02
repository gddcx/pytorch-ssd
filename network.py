import torch
import torch.nn as nn
import torchvision


class Backbone(nn.Module):
    def __init__(self, pretrain=False):
        super().__init__()
        features = torchvision.models.vgg16(pretrained=pretrain).features
        features = list(features.children())
        features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # pool5
        features.extend([
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1, stride=1), # conv6
            nn.ReLU()
        ])
        features.extend([
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1, padding=0, stride=1), # conv7
            nn.ReLU()
        ])
        self.subnet1 = nn.Sequential(*features[:23])
        self.subnet2 = nn.Sequential(*features[23:])
        self.subnet3 = nn.Sequential(  # conv8
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.subnet4 = nn.Sequential( # conv9
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.subnet5 = nn.Sequential( # conv10
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )
        self.subnet6 = nn.Sequential( # conv11
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=0)
        )

    def forward(self, data):
        out1 = self.subnet1(data) # TODO: Conv4-3要L2处理尺度问题
        out2 = self.subnet2(out1)
        out3 = self.subnet3(out2)
        out4 = self.subnet4(out3)
        out5 = self.subnet5(out4)
        out6 = self.subnet6(out5)
        return out1, out2, out3, out4, out5, out6


class Detection(nn.Module):
    def __init__(self, num_category):
        super().__init__()
        # channels 512, 1024, 512, 256, 256, 256
        self.num_category = num_category
        self.net = nn.ModuleList()
        self.net.append(nn.Conv2d(in_channels=512, out_channels=4 * 4 + 4 * num_category, kernel_size=3, stride=1, padding=1))
        self.net.append(nn.Conv2d(in_channels=1024, out_channels=6 * 4 + 6 * num_category, kernel_size=3, stride=1, padding=1))
        self.net.append(nn.Conv2d(in_channels=512, out_channels=6 * 4 + 6 * num_category, kernel_size=3, stride=1, padding=1))
        self.net.append(nn.Conv2d(in_channels=256, out_channels=6 * 4 + 6 * num_category, kernel_size=3, stride=1, padding=1))
        self.net.append(nn.Conv2d(in_channels=256, out_channels=4 * 4 + 4 * num_category, kernel_size=3, stride=1, padding=1))
        self.net.append(nn.Conv2d(in_channels=256, out_channels=4 * 4 + 4 * num_category, kernel_size=3, stride=1, padding=1))

    def forward(self, data):
        out_list = []
        batch_size = data[0].shape[0]
        for idx in range(len(self.net)):
            out = self.net[idx](data[idx])
            out = out.permute(0, 2, 3, 1) # bs, fms, fms, dbn*(4+num_category)
            out = out.reshape(batch_size, -1, 4+self.num_category)
            out_list.append(out)
        return torch.cat(out_list, dim=1)


class SSDNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone(pretrain=True)
        self.detection = Detection(num_category=21)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        out = self.backbone(data)
        out = self.detection(out)
        return out
