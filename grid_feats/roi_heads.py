# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    ROIHeads,
    Res5ROIHeads,
    StandardROIHeads,
)


class AttributePredictor(nn.Module):
	"""
    Head for attribute prediction, including feature/score computation and
    loss computation.

    """
    def __init__(self, cfg, input_dim):
        super().__init__()

        # fmt: off
        self.num_objs          = cfg.MODEL.ROI_HEADS.NUM_CLASSES
        self.obj_embed_dim     = cfg.MODEL.ROI_ATTRIBUTE_HEAD.OBJ_EMBED_DIM
        self.fc_dim            = cfg.MODEL.ROI_ATTRIBUTE_HEAD.FC_DIM
        self.num_attributes    = cfg.MODEL.ROI_ATTRIBUTE_HEAD.NUM_CLASSES
        self.max_attr_per_ins  = cfg.INPUT.MAX_ATTR_PER_INS
        self.loss_weight       = cfg.MODEL.ROI_ATTRIBUTE_HEAD.LOSS_WEIGHT
        # fmt: on

        # object class embedding, including the background class
        self.obj_embed = nn.Embedding(self.num_objs + 1, self.obj_embed_dim)
        input_dim += self.obj_embed_dim
        self.fc = nn.Sequential(
                nn.Linear(input_dim, self.fc_dim),
                nn.ReLU()
            )
        self.attr_score = nn.Linear(self.fc_dim, self.num_attributes)
        nn.init.normal_(self.attr_score.weight, std=0.01)
        nn.init.constant_(self.attr_score.bias, 0)

    def forward(self, x, obj_labels):
        attr_feat = torch.cat((x, self.obj_embed(obj_labels)), dim=1)
        return self.attr_score(self.fc(attr_feat))

    def loss(self, score, label):
        n = score.shape[0]
        score = score.unsqueeze(1)
        score = score.expand(n, self.max_attr_per_ins, self.num_attributes).contiguous()
        score = score.view(-1, self.num_attributes)
        inv_weights = (
            (label >= 0).sum(dim=1).repeat(self.max_attr_per_ins, 1).transpose(0, 1).flatten()
        )
        weights = inv_weights.float().reciprocal()
        weights[weights > 1] = 0.
        n_valid = len((label >= 0).sum(dim=1).nonzero())
        label = label.view(-1)
        attr_loss = F.cross_entropy(score, label, reduction="none", ignore_index=-1)
        attr_loss = (attr_loss * weights).view(n, -1).sum(dim=1)

        if n_valid > 0:
            attr_loss = attr_loss.sum() * self.loss_weight / n_valid
        else:
            attr_loss = attr_loss.sum() * 0.
        return {"loss_attr": attr_loss}


class AttributeROIHeads(ROIHeads):
	"""
    An extension of ROIHeads to include attribute prediction.
    """
    def forward_attribute_loss(self, proposals, box_features):
        proposals, fg_selection_attributes = select_foreground_proposals(
            proposals, self.num_classes
        )
        attribute_features = box_features[torch.cat(fg_selection_attributes, dim=0)]
        obj_labels = torch.cat([p.gt_classes for p in proposals])
        attribute_labels = torch.cat([p.gt_attributes for p in proposals], dim=0)
        attribute_scores = self.attribute_predictor(attribute_features, obj_labels)
        return self.attribute_predictor.loss(attribute_scores, attribute_labels)


@ROI_HEADS_REGISTRY.register()
class AttributeRes5ROIHeads(AttributeROIHeads, Res5ROIHeads):
	"""
    An extension of Res5ROIHeads to include attribute prediction.
    """
    def __init__(self, cfg, input_shape):
    	super(Res5ROIHeads, self).__init__(cfg, input_shape)

    	assert len(self.in_features) == 1

        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        self.attribute_on = cfg.MODEL.ATTRIBUTE_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON

        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.res5, out_channels = self._build_res5_block(cfg)
        self.box_predictor = FastRCNNOutputLayers(
            out_channels, self.num_classes, self.cls_agnostic_bbox_reg
        )

        if self.mask_on:
            self.mask_head = build_mask_head(
                cfg,
                ShapeSpec(channels=out_channels, width=pooler_resolution, height=pooler_resolution),
            )

        if self.attribute_on:
            self.attribute_predictor = AttributePredictor(cfg, out_channels)

    def forward(self, images, features, proposals, targets=None):
        del images

        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])
        pred_class_logits, pred_proposal_deltas = self.box_predictor(feature_pooled)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            del features
            losses = outputs.losses()

            if self.mask_on:
                proposals, fg_selection_masks = select_foreground_proposals(
                    proposals, self.num_classes
                )
                mask_features = box_features[torch.cat(fg_selection_masks, dim=0)]
                del box_features
                losses.update(self.mask_head(mask_features, proposals))

            if self.attribute_on:
                losses.update(self.forward_attribute_loss(proposals, box_features))

            return [], losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class AttributeStandardROIHeads(AttributeROIHeads, StandardROIHeads):
	"""
    An extension of StandardROIHeads to include attribute prediction.
    """
    def __init__(self, cfg, input_shape):
    	StandardROIHeads.__init__(cfg, input_shape)

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        self.attribute_on        = cfg.MODEL.ATTRIBUTE_ON
        # fmt: on

        in_channels = [input_shape[f].channels for f in self.in_features]
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        self.box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )
        self.box_predictor = FastRCNNOutputLayers(
            self.box_head.output_size, self.num_classes, self.cls_agnostic_bbox_reg
        )

        if self.attribute_on:
            self.attribute_predictor = AttributePredictor(cfg, self.box_head.output_size)

    def _forward_box(self, features, proposals):
        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            losses = outputs.losses()

            if self.attribute_on:
                losses.update(self.forward_attribute_loss(proposals, box_features))

            return losses
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances
