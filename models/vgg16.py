import torch
import torch.nn as nn


# class N_conv(nn.Module):
#     """
#     This a class for defining the N filters
#     Attributes
#     ----------
#     conv  : nn.Sequential
#         defines the train model
#     """

#     def __init__(self, in_channels, out_channels, N=2):
#         super().__init__()
#         model = []
#         model.append(
#             nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1))
#         )
#         model.append(nn.BatchNorm2d(out_channels))
#         model.append(nn.ReLU())

#         for _ in range(N - 1):
#             model.append(
#                 nn.Conv2d(
#                     out_channels, out_channels, kernel_size=(3, 3), padding=(1, 1)
#                 )
#             )
#             model.append(nn.BatchNorm2d(out_channels))
#             model.append(nn.ReLU())

#         model.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
#         self.conv = nn.Sequential(*model)

#     def forward(self, x):
#         return self.conv(x)


# class VGG16(nn.Module):
#     def __init__(self, num_classes: int = 10, init_weights: bool = True):
#         super().__init__()
#         # convolutional layers (feature extraction)
#         self.conv1 = N_conv(3, 64)
#         self.conv2 = N_conv(64, 128)
#         self.conv3 = N_conv(128, 256, N=3)
#         self.conv4 = N_conv(256, 512, N=3)
#         self.conv5 = N_conv(512, 512, N=3)
#         self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

#         if init_weights:
#             self._initialize_weights()

#         # Fully connected layers (classifier)
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(4096, 4096),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(4096, num_classes),
#         )

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.normal_(m.weight, 0, 0.01)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = self.conv4(x)
#         x = self.conv5(x)
#         x = self.avgpool(x)
#         x = self.classifier(x)
#         return x


"""VGG11/13/16/19 in Pytorch."""
import torch
import torch.nn as nn


cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        "M",
    ],
    "VGG19": [
        64,
        64,
        "M",
        128,
        128,
        "M",
        256,
        256,
        256,
        256,
        "M",
        512,
        512,
        512,
        512,
        "M",
        512,
        512,
        512,
        512,
        "M",
    ],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True),
                ]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def test():
    net = VGG("VGG11")
    x = torch.randn(2, 3, 32, 32)
    y = net(x)
    print(y.size())
