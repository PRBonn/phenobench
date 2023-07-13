from typing import Dict, List, Tuple

import torch

IMG_WIDTH = 1024
IMG_HEIGHT = 1024

def convert_partial_semantics(semantics: torch.Tensor) -> torch.Tensor:
  semantics_converted = semantics.clone()

  # convert partial crops to regular crops
  mask_partial_crops = semantics == 3
  semantics_converted[mask_partial_crops] = 1

  # convert partial weeds to regular weeds
  mask_partial_weeds = semantics == 4
  semantics_converted[mask_partial_weeds] = 2

  return semantics_converted

def bounding_box_from_mask_xyxy(mask: torch.Tensor) -> Tuple[int, int, int, int]:
  rows, cols = mask.nonzero(as_tuple=True)

  x_min = int(torch.min(cols))
  y_min = int(torch.min(rows))
  x_max = int(torch.max(cols))
  y_max = int(torch.max(rows))

  return x_min, y_min, x_max, y_max

def bounding_box_from_mask_cxcywh(mask: torch.Tensor) -> Tuple[int, int, int, int]:

  x_min, y_min, x_max, y_max = bounding_box_from_mask_xyxy(mask)

  cx = int(round((x_min + x_max) / 2))
  cy = int(round((y_min + y_max) / 2))
  width = x_max - x_min
  height = y_max - y_min

  assert width >= 0
  assert height >= 0

  return cx, cy, width, height

def bounding_box_from_cxcywh_to_xyxy(cx: int, cy: int, width: int, height: int) -> Tuple[int, int, int, int]:
  x_min = round(cx - (width / 2))
  x_max = round(cx + (width / 2))
  y_min = round(cy - (height / 2))
  y_max = round(cy + (height / 2))

  return x_min, x_max, y_min, y_max

def box_to_mask(x_min:int, x_max: int, y_min: int, y_max: int) -> torch.Tensor:
  canvas = torch.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=torch.bool)

  canvas[y_min: (y_max + 1), x_min: (x_max + 1)] = True

  return canvas

def cvt_yolo_to_bbox_map(yolo_anno: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
  """ Convert YOLO annotations to format required by MAP of torchmetrics.

  Please check the following to get more information: https://torchmetrics.readthedocs.io/en/stable/detection/mean_average_precision.html

  Args:
      yolo_anno (torch.Tensor): annotations provided by YOLO ... of shape [n_preds x 6]
        - 1st col -> label id
        - 2nd col -> center_x in [0,1]
        - 3rd col -> center_y in [0,1]
        - 4th col -> width in [0,1]
        - 5th col -> height in [0,1]
        - 6th col -> confidence score in [0,1]

  Returns:
      List[Dict[str, torch.Tensor]]: format required by MAP of torchmetrics.
  """

  if len(yolo_anno.shape) == 1:
    yolo_anno = yolo_anno.unsqueeze(dim=0)

  n_predicitions = yolo_anno.shape[0]
  boxes = []
  labels = []
  scores = []
  for i in range(n_predicitions):
    # ------- Label -------
    try:
      label = int(yolo_anno[i, 0]) # + 1
    except IndexError:
      continue
    if label == 0:
      continue

    # ------- Bounding Box -------
    cx = int(round(float(yolo_anno[i, 1]) * IMG_WIDTH))
    cy = int(round(float(yolo_anno[i, 2]) * IMG_HEIGHT))
    width = int(round(float(yolo_anno[i, 3]) * IMG_WIDTH))
    height = int(round(float(yolo_anno[i, 4]) * IMG_HEIGHT))

    # ------- Confidence Score -------
    score = float(yolo_anno[i, 5])

    # ------- Accumulate -------
    labels.append(torch.Tensor([label]))
    boxes.append(torch.Tensor([cx, cy, width, height]))
    scores.append(torch.Tensor([score]))

  out = {}
  try:
    out['labels'] = torch.stack(labels).squeeze(1).type(torch.uint8)  # [n_objects]
    out['boxes'] = torch.stack(boxes)  # [n_objects x 4]
    out['scores'] = torch.stack(scores).squeeze(1)  # [n_objects]
  except RuntimeError:
    # i.e., there isn't any prediction
    out['labels'] = torch.tensor([])
    out['boxes'] = torch.tensor([])
    out['scores'] = torch.tensor([])


  return [out]


def cvt_gt_to_bbox_map(instance_map: torch.Tensor, semantics: torch.Tensor, visibility: torch.Tensor) -> List[Dict[str, torch.Tensor]]:
  assert torch.max(visibility) <= 1.0
  assert torch.min(visibility) >= 0.0

  instance_ids = torch.unique(instance_map)
  instance_ids = instance_ids[instance_ids != 0]  # ignore background

  boxes = []
  labels = []
  visibilities = []
  for instance_id in instance_ids:
    # ------- Mask -------
    instance_mask = instance_map == instance_id

    # ------- Bounding Box -------
    cx, cy, width, height = bounding_box_from_mask_cxcywh(instance_mask)

    # ------- Label -------
    label = torch.unique(semantics[instance_mask])
    try:
      if label == 0:
        continue
      if len(label) != 1:
        assert False, "Single instance has multiple semantic classes"
    except RuntimeError:
      continue

    # ------- Visibility -------
    vis = torch.unique(visibility[instance_mask])

    # ------- Accumulate -------
    boxes.append(torch.Tensor([cx, cy, width, height]))
    labels.append(torch.Tensor([label]))
    visibilities.append(torch.Tensor(vis))

  out = {}
  out['labels'] = torch.stack(labels).squeeze(1).type(torch.uint8)  # [n_objects]
  out['boxes'] = torch.stack(boxes)  # [n_objects x 4]
  out['visibility'] = torch.stack(visibilities)  # [n_objects]

  return [out]
