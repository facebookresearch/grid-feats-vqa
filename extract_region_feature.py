#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
Region features extraction script.
"""
import argparse
import os
import torch
import tqdm
from fvcore.common.file_io import PathManager

import numpy as np
from torch.nn import functional as F

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_setup
from detectron2.evaluation import inference_context
from detectron2.modeling import build_model
from detectron2.modeling import postprocessing

from grid_feats import (
    add_attribute_config,
    build_detection_test_loader_with_attributes,
    build_detection_test_loader_for_images,
)

# A simple mapper from object detection dataset to VQA dataset names
dataset_to_folder_mapper = {}
dataset_to_folder_mapper["coco_2014_train"] = "train2014"
dataset_to_folder_mapper["coco_2014_val"] = "val2014"
# One may need to change the Detectron2 code to support coco_2015_test
# insert "coco_2015_test": ("coco/test2015", "coco/annotations/image_info_test2015.json"),
# at: https://github.com/facebookresearch/detectron2/blob/master/detectron2/data/datasets/builtin.py#L36
dataset_to_folder_mapper["coco_2015_test"] = "test2015"


def extract_feature_argument_parser():
    parser = argparse.ArgumentParser(description="Region feature extraction")
    parser.add_argument(
        "--config-file", default="", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--dataset",
        help="name of the dataset",
        default="coco_2014_train",
        choices=["coco_2014_train", "coco_2014_val", "coco_2015_test"],
    )
    parser.add_argument(
        "--dataset-path",
        help="path to image folder dataset, if not provided expects a detection dataset",
        default="",
    )
    parser.add_argument(
        "--feature-name",
        help="type of feature to extract, for FPN and DC5 models it can be either of the last two FC regression outputs",
        default="fc7",
        choices=["fc6", "fc7"],
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


def extract_feature_on_dataset(model, data_loader, dump_folder, feature_name):
    for idx, inputs in enumerate(tqdm.tqdm(data_loader)):
        with torch.no_grad():
            image_id = inputs[0]["image_id"]

            # compute features and proposals
            images = model.preprocess_image(inputs)
            features = model.backbone(images.tensor)
            proposals, _ = model.proposal_generator(images, features)

            # pooled features and box predictions
            _, pooled_features, pooled_features_fc6 = model.roi_heads.get_roi_features(
                features, proposals
            )
            predictions = model.roi_heads.box_predictor(pooled_features)
            cls_probs = F.softmax(predictions[0], dim=-1)
            cls_probs = cls_probs[:, :-1]  # background is last
            predictions, r_indices = model.roi_heads.box_predictor.inference(
                predictions, proposals
            )
            # Create Boxes objects from proposals. Since features are extrracted from
            # the proposal boxes we use them instead of predicted boxes.
            box_type = type(proposals[0].proposal_boxes)
            proposal_bboxes = box_type.cat([p.proposal_boxes for p in proposals])
            proposal_bboxes.tensor = proposal_bboxes.tensor[r_indices]
            predictions[0].set("proposal_boxes", proposal_bboxes)
            predictions[0].remove("pred_boxes")

            # postprocess
            height = inputs[0].get("height")
            width = inputs[0].get("width")
            r = postprocessing.detector_postprocess(predictions[0], height, width)

            bboxes = r.get("proposal_boxes").tensor
            classes = r.get("pred_classes")
            cls_probs = cls_probs[r_indices]
            if feature_name == "fc6" and pooled_features_fc6 is not None:
                pooled_features = pooled_features_fc6[r_indices]
            else:
                pooled_features = pooled_features[r_indices]

            assert (
                bboxes.size(0)
                == classes.size(0)
                == cls_probs.size(0)
                == pooled_features.size(0)
            )

            # save info and features
            info = {
                "bbox": bboxes.cpu().numpy(),
                "num_boxes": bboxes.size(0),
                "objects": classes.cpu().numpy(),
                "image_height": r.image_size[0],
                "image_width": r.image_size[1],
                "cls_prob": cls_probs.cpu().numpy(),
                "features": pooled_features.cpu().numpy()
            }
            np.save(os.path.join(dump_folder, str(image_id) + ".npy"), info)


def do_feature_extraction(cfg, model, dataset_name, dataset_path, feature_name):
    with inference_context(model):
        dump_folder = os.path.join(
            cfg.OUTPUT_DIR, "features", dataset_to_folder_mapper[dataset_name]
        )
        PathManager.mkdirs(dump_folder)
        if dataset_path is not "":
            data_loader = build_detection_test_loader_for_images(cfg, dataset_path)
        else:
            data_loader = build_detection_test_loader_with_attributes(cfg, dataset_name)

        extract_feature_on_dataset(model, data_loader, dump_folder, feature_name)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_attribute_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    do_feature_extraction(cfg, model, args.dataset, args.dataset_path, args.feature_name)


if __name__ == "__main__":
    args = extract_feature_argument_parser().parse_args()
    print("Command Line Args:", args)
    main(args)
