from typing import Dict

import torch

from .convert import bounding_box_from_cxcywh_to_xyxy, box_to_mask


def filter_partial_masks(pred_instance_masks: torch.Tensor, pred_semantics: torch.Tensor, gt_instance_masks: torch.Tensor, gt_semantics: torch.Tensor, gt_visibility: torch.Tensor):
  assert torch.max(gt_visibility) <= 1.0
  assert torch.min(gt_visibility) >= 0.0

  gt_instance_partial_masks = []
  # set semantics of partial masks to 0 and store partial_masks
  for gt_instance_id in torch.unique(gt_instance_masks):
    if gt_instance_id == 0:
      continue
    gt_mask = gt_instance_masks == gt_instance_id
    gt_vis = torch.unique(gt_visibility[gt_mask]) # contain a single value
    if gt_vis > 0.5:
      continue
    gt_semantics[gt_mask] = 0
    gt_instance_partial_masks.append(gt_mask)


  for pred_instance_id in torch.unique(pred_instance_masks):
    if pred_instance_id == 0:
      continue
    pred_mask = pred_instance_masks == pred_instance_id

    for gt_mask in gt_instance_partial_masks:
      # compute how much of the prediction is within the ground truth
      a = torch.sum(gt_mask & pred_mask)
      b = torch.sum(pred_mask)
      score = a / (b + 1e-12)
      assert score <= 1.0

      if score > 0.5:
        pred_semantics[pred_mask] = 0

def filter_partials_boxes(preds: Dict, targets: Dict):
  n_preds = preds['boxes'].shape[0]
  n_targets = targets['boxes'].shape[0]

  remove_pred_indicies = []
  remove_gt_indicies = []

  for i in range(n_preds):
    bbox_pred = preds['boxes'][i, :]
    cx, cy, width, height = bbox_pred[0].item(), bbox_pred[1].item(), bbox_pred[2].item(), bbox_pred[3].item()
    xmin, xmax, ymin, ymax = bounding_box_from_cxcywh_to_xyxy(cx, cy, width, height)
    bbox_pred_canvas = box_to_mask(xmin, xmax, ymin, ymax)

    for j in range(n_targets):
      vis = targets['visibility'][j]
      if vis > 0.5:
        continue
      remove_gt_indicies.append(j)

      bbox_target = targets['boxes'][j, :]
      cx, cy, width, height = bbox_target[0].item(), bbox_target[1].item(), bbox_target[2].item(), bbox_target[3].item()
      xmin, xmax, ymin, ymax = bounding_box_from_cxcywh_to_xyxy(cx, cy, width, height)
      bbox_gt_canvas = box_to_mask(xmin, xmax, ymin, ymax)

      # compute how much of the prediction is within the ground truth
      a = torch.sum(bbox_pred_canvas & bbox_gt_canvas)
      b = torch.sum(bbox_pred_canvas)
      score = a / (b + 1e-12)
      assert score <= 1.0

      if score > 0.5:
        remove_pred_indicies.append(i)

  # filter predictions
  mask_pred = torch.ones_like(preds['labels'], dtype=torch.bool)
  for rm_idx in remove_pred_indicies:
    mask_pred[rm_idx] = False
  preds['labels'] = preds['labels'][mask_pred]
  preds['scores'] = preds['scores'][mask_pred]
  preds['boxes'] = preds['boxes'][mask_pred]

  # filter targets
  mask_target = torch.ones_like(targets['labels'], dtype=torch.bool)
  for rm_idx in remove_gt_indicies:
    mask_target[rm_idx] = False
  targets['labels'] = targets['labels'][mask_target]
  targets['boxes'] = targets['boxes'][mask_target]
  targets['visibility'] = targets['visibility'][mask_target]
