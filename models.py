"""The networks.

- alexnet:   the network used in the AffectNet paper
- resnet18:  a newer, smaller and usually more accurate network (default)
- small_cnn: a compact CNN trained from scratch, to show how much ImageNet pretraining helps

In each case the ImageNet classifier is replaced with a new layer that outputs one score per expression.
"""
import torch
import torch.nn as nn
import torchvision


def build_model(name, num_classes, pretrained=True, face_recognition_checkpoint=False):
    tvm = torchvision.models
    if name == "resnet18":
        model = tvm.resnet18(weights=tvm.ResNet18_Weights.DEFAULT if pretrained else None)
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, num_classes))
        if face_recognition_checkpoint:
            model_state_dict = torch.load("epoch_16_acc_0.624.pt")["model"]
            model.load_state_dict(model_state_dict, strict=False)
    elif name == "alexnet":
        model = tvm.alexnet(weights=tvm.AlexNet_Weights.DEFAULT if pretrained else None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif name == "small_cnn":
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.MaxPool2d(2))
        model = nn.Sequential(block(3, 64), block(64, 128), block(128, 256), block(256, 512),
                              nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.4), nn.Linear(512, num_classes))
    else:
        raise ValueError(f"Unknown model: {name}")
    return model
