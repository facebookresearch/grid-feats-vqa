#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Grid features extraction script using torchvision.
"""
import argparse
from collections import OrderedDict
import numpy as np
import re

import torch
import torch.nn as nn
import torchvision
from fvcore.common.file_io import PathManager

from detectron2.layers import FrozenBatchNorm2d
from detectron2.data.detection_utils import read_image
from detectron2.data import transforms as T


def extract_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Grid feature extraction using torchvision")
    parser.add_argument("--backbone", 
                        default="X-152", 
                        help="backbone of the network")
    parser.add_argument("--weights", 
                        metavar="FILE", 
                        help="path to the weights file", 
                        default="./X-152.pth")
    parser.add_argument("--image", 
                        metavar="FILE", 
                        help="path to the image file", 
                        default="./image.jpg")
    parser.add_argument("--output", 
                        metavar="FILE", 
                        help="path to the output file that saves features", 
                        default="./output.pth")
    return parser


class GridFeaturesTorchVision(torchvision.models.ResNet):
    def __init__(self, _type="R-50", detectron_weights=None):
        kwargs = dict()
        kwargs["norm_layer"] = FrozenBatchNorm2d
        if _type == "R-50":
            blocks = [3, 4, 6, 3]
        elif _type == "X-101":
            kwargs["groups"] = 32
            kwargs["width_per_group"] = 8
            blocks = [3, 4, 23, 3]
        elif _type == "X-152":
            kwargs["groups"] = 32
            kwargs["width_per_group"] = 8
            blocks = [3, 8, 36, 3]

        super().__init__(torchvision.models.resnet.Bottleneck, blocks, **kwargs)

        # Remove FC and avgpool
        del self.fc
        del self.avgpool

        if detectron_weights is not None:
            # Need to load before adding mean/std keys
            detectron_state = torch.load(detectron_weights, map_location="cpu")
            self.load_state_dict(self.replace_detectron_keys(detectron_state["model"]),
                                 strict=True)

        self.register_buffer(
            "_mean",
            torch.tensor([103.530, 116.280, 123.675], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )
        self.register_buffer(
            "_std",
            torch.tensor([57.375, 57.120, 58.395], dtype=torch.float32).view(
                1, 3, 1, 1
            ),
        )

        # ResNet50 uses stride in the 1x1 in detectron2
        if _type == "R-50":
            # Uses stride
            for l in [self.layer2, self.layer3, self.layer4]:
                l[0].conv1.stride = (2, 2)
                l[0].conv2.stride = (1, 1)
        else:
            for l in [self.layer2, self.layer3, self.layer4]:
                l[0].conv1.stride = (1, 1)
                l[0].conv2.stride = (2, 2)

        # The ResNet50 and ResNeXt152 had the
        # std fused into the conv weights, so replace with 1s
        if _type in {"R-50", "X-152"}:
            self._std.fill_(1.0)

    @staticmethod
    def replace_module(groups):
        return "bn{}".format(groups[1])

    @classmethod
    def replace_detectron_keys(cls, pretrained_state_dict):
        new_pretrained_state_dict = OrderedDict()
        replace_shortcut = {
            "shortcut.weight": "downsample.0.weight",
            "shortcut.norm": "downsample.1",
        }
        replace_layer = {
            "stem.": "",
            "res2": "layer1",
            "res3": "layer2",
            "res4": "layer3",
            "res5": "layer4",
        }
        blacklist = {"proposal_generator", "roi_heads"}

        for key, value in pretrained_state_dict.items():
            if any(key.startswith(ele) for ele in blacklist):
                continue

            new_key = re.sub("backbone\.", "", key)
            new_key = re.sub(
                r"(stem\.|res2|res3|res4|res5)",
                lambda x: replace_layer[x.groups()[0]],
                new_key,
            )
            new_key = re.sub(
                r"(shortcut[.]weight|shortcut[.]norm)",
                lambda x: replace_shortcut[x.groups()[0]],
                new_key,
            )
            new_key = re.sub(
                r"(conv)([1-3])(.norm)",
                lambda x: cls.replace_module(x.groups()),
                new_key,
            )
            new_pretrained_state_dict[new_key] = value

        return new_pretrained_state_dict

    def forward(self, x):
        x = (x - self._mean) / self._std

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x


if __name__ == "__main__":
    args = extract_feature_argument_parser().parse_args()
    print("Command Line Args:", args)
    # build network
    net = GridFeaturesTorchVision(args.backbone, 
                                  detectron_weights=args.weights)
    # read and prepare images
    image = read_image(args.image, format="BGR")
    tfm_gens = [T.ResizeShortestEdge(600, 1000, "choice")]
    image, _ = T.apply_transform_gens(tfm_gens, image)
    image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
    # apply the network
    res = net(image.unsqueeze(0))
    # save features
    with PathManager.open(args.output, "wb") as f:
        # save as CPU tensors
        torch.save(res.cpu(), f)
