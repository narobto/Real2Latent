# import torch
# import torch.nn as nn
# import torchvision
#
#
# def disabled_train(self, mode=True):
#     """Overwrite model.train with this function to make sure train/eval mode
#     does not change anymore."""
#     return self
#
# class MessageDecoder(nn.Module):
#     def __init__(self, message_len=48, pretrained_weights= None):
#         super().__init__()
#
#         if pretrained_weights != None:
#             self.decoder = torchvision.models.resnet50(pretrained=False, progress=False)
#             checkpoint = torch.load(pretrained_weights)
#             self.decoder.load_state_dict(checkpoint)
#             self.decoder.fc = nn.Linear(self.decoder.fc.in_features, message_len)
#         else:
#             self.decoder = torchvision.models.resnet50(pretrained=True, progress=False)
#             self.decoder.fc = nn.Linear(self.decoder.fc.in_features, message_len)
#
#     def forward(self, image):
#         x = self.decoder(image)
#         return x






import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)



class RefinementBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(RefinementBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.se = SEBlock(out_channels)

    def forward(self, x):
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out


class MessageDecoder(nn.Module):
    def __init__(self, message_len=48, pretrained_weights=None, hidden_dim=256):

        super(MessageDecoder, self).__init__()


        if pretrained_weights is not None:
            print(f"Loading backbone weights from {pretrained_weights}")
            backbone = models.resnet50(pretrained=False)
            checkpoint = torch.load(pretrained_weights)
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            checkpoint = {k: v for k, v in checkpoint.items() if 'fc' not in k}
            backbone.load_state_dict(checkpoint, strict=False)
        else:
            backbone = models.resnet50(pretrained=True)

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer1 = backbone.layer1  # [B, 256, H/4, W/4]  <- 核心层
        self.layer2 = backbone.layer2  # [B, 512, H/8, W/8]
        self.layer3 = backbone.layer3  # [B, 1024, H/16, W/16]
        self.layer4 = backbone.layer4  # [B, 2048, H/32, W/32]

        self.layer1_dim = 512
        self.lat_layer1 = RefinementBlock(256, self.layer1_dim)


        self.lat_layer2 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.lat_layer3 = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        self.lat_layer4 = nn.Conv2d(2048, hidden_dim, kernel_size=1)


        total_concat_dim = self.layer1_dim + hidden_dim * 3

        self.head = nn.Sequential(
            nn.Linear(total_concat_dim, 1024),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(1024, message_len)
        )

    def forward(self, image):

        x = self.conv1(image)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        c2 = self.layer1(x)  # H/4 (包含 Latent 和 Level 0/1 的高频信息)
        c3 = self.layer2(c2)  # H/8
        c4 = self.layer3(c3)  # H/16
        c5 = self.layer4(c4)  # H/32


        p2 = self.lat_layer1(c2)

        p3 = self.lat_layer2(c3)
        p4 = self.lat_layer3(c4)
        p5 = self.lat_layer4(c5)


        out2 = F.adaptive_avg_pool2d(p2, (1, 1)).flatten(1)  # [B, 512]
        out3 = F.adaptive_avg_pool2d(p3, (1, 1)).flatten(1)  # [B, 256]
        out4 = F.adaptive_avg_pool2d(p4, (1, 1)).flatten(1)  # [B, 256]
        out5 = F.adaptive_avg_pool2d(p5, (1, 1)).flatten(1)  # [B, 256]


        features = torch.cat([out2, out3, out4, out5], dim=1)  # [B, 1280]


        out = self.head(features)
        return out



