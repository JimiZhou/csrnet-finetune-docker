"""Standalone CSRNet model for fine-tuning."""

import torch
import torch.nn as nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, load_weights: bool = False):
        super().__init__()
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.frontend = self._make_layers(self.frontend_feat)
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = self._make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if load_weights:
            self._initialize_weights()
            self._load_vgg_weights()

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x

    def _make_layers(self, cfg, in_channels=3, batch_norm=False, dilation=False):
        d_rate = 2 if dilation else 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _load_vgg_weights(self):
        try:
            from torchvision.models import VGG16_Weights
            vgg = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
            vgg_layers = list(vgg.features.children())
            frontend_layers = list(self.frontend.children())

            vgg_idx = 0
            frontend_idx = 0
            while vgg_idx < len(vgg_layers) and frontend_idx < len(frontend_layers):
                vgg_layer = vgg_layers[vgg_idx]
                frontend_layer = frontend_layers[frontend_idx]

                if isinstance(vgg_layer, nn.Conv2d) and isinstance(frontend_layer, nn.Conv2d):
                    frontend_layer.weight.data = vgg_layer.weight.data.clone()
                    if vgg_layer.bias is not None and frontend_layer.bias is not None:
                        frontend_layer.bias.data = vgg_layer.bias.data.clone()
                    vgg_idx += 1
                    frontend_idx += 1
                elif isinstance(vgg_layer, type(frontend_layer)):
                    vgg_idx += 1
                    frontend_idx += 1
                else:
                    vgg_idx += 1
        except Exception as exc:
            print(f"Warning: Could not load VGG-16 weights: {exc}")

