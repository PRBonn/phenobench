from typing import Dict, List, Tuple

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from PIL import Image


def draw_semantics(axes: Axes,
                   image: Image,
                   semantics: np.array,
                   alpha: float = 0.5,
                   colors: Dict[int, Tuple[int, int, int]] = {
                       1: (0, 255, 0),
                       2: (255, 0, 0),
                       3: (0, 255, 255),
                       4: (255, 0, 255)
                   },
                   mask_classes: List[int] = [0]):
  """ draw pixel-wise semantics according to given image and semantics. 

  Args:
      axes (matplotlib.axes.Axes): axes object to draw into.
      image (PIL.Image): image that should be drawn.
      semantics (np.array): semantics (1=crop, 2=weed, 3=partial crop, 4=partial weed)
      alpha(float): transparency of semantic mask drawn on the image.
      colors(Dict[int, tuple]): mapping of class ids to colors. Default: 1(crop) = (0,255,0), 2(weed) = (0,255,0), 3(partial_crop)= (0,255,255), 4(partial_weed) = (255,0,255)
      mask_classes (List[int]): class indices that are not shown in the final composition.
  """

  im_semantics = np.zeros((*semantics.shape, 3), dtype=float)

  for label, color in colors.items():
    im_semantics[semantics == label] = np.array(color) / 255.0

  im_alpha = np.ones((*semantics.shape, 3), dtype=float) * alpha
  for c in mask_classes:
    im_alpha[semantics == c] = 0.0

  composed_image = (1.0 - im_alpha) * (np.array(image) / 255.0) + im_alpha * im_semantics
  axes.imshow(composed_image)

  # not sure why, but `ax.im_show(im_semantics, alpha=im_alpha)` doesn't work as expected:
  # im_alpha = np.ones(semantics.shape, dtype=float) * alpha
  # for c in mask_classes: im_alpha[semantics == c] = 0.0
  # axes.imshow(image)
  # axes.imshow(im_semantics, alpha = im_alpha)


def draw_instances(ax: Axes, image: Image, instance_mask: np.array, alpha: float = 0.5):
  """ Draw instance masks from provided data, where individual masks are indicated by random colors.

  Args:
      ax (Axes): axes object to draw into.
      image (Image): image that should be drawn.
      instance_masks (np.array): _description_
      alpha (float, optional): _description_. Defaults to 0.5.
  """
  random_colors = plt.get_cmap('tab10').resampled(10)
  unique_ids = np.unique(instance_mask)
  # unique_ids = unique_ids[unique_ids > 0]

  im_mask = np.zeros((*instance_mask.shape, 3))
  for idx, uid in enumerate(unique_ids):
    im_mask[instance_mask == uid] = random_colors(idx % 10)[:3]

  im_alpha = np.ones((*instance_mask.shape, 3)) * alpha
  im_alpha[instance_mask == 0, :] = 0.0

  float_image = (np.array(image) / 255.0)
  composed_image = (1.0 - im_alpha) * float_image + im_alpha * im_mask

  ax.imshow(composed_image)


def draw_bboxes(ax: Axes,
                image: Image,
                bboxes: List[Dict],
                colors: Dict[int, Tuple[int, int, int]] = {
                    1: (0, 255, 0),
                    2: (255, 0, 0),
                    3: (0, 255, 255),
                    4: (255, 0, 255)
                },
                linewidth=2):
  """ Draw the given bounding boxes on the image.

  Args:
      ax (Axes): axes object to draw into.
      image (Image): image that should be drawn.
      bboxes (List[Dict): list of bounding boxes in the format {"label":<label>, "center": (x,y), "width": w, "height": h},
      colors(Dict[int, tuple]): mapping of class ids to colors. Default: 1(crop) = (0,255,0), 2(weed) = (0,255,0), 3(partial_crop)= (0,255,255), 4(partial_weed) = (255,0,255)
      linewidth(int): thickness of the lines of the bounding box. Default: 2
  """

  ax.imshow(image)

  for bbox in bboxes:
    x, y = bbox["center"][0] - bbox["width"] / 2, bbox["center"][1] - bbox["height"] / 2
    rect = patches.Rectangle((x, y),
                             bbox["width"],
                             bbox["height"],
                             linewidth=linewidth,
                             edgecolor=np.array(colors[bbox["label"]]) / 255.0,
                             facecolor='none')
    ax.add_patch(rect)
