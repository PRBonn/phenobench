#!/usr/bin/env python3

import argparse
from pathlib import Path

from phenobench.evaluation.auxiliary.common import centered_text
from phenobench.evaluation.evaluate_leaf_bounding_boxes import evaluate_leaf_detection
from phenobench.evaluation.evaluate_leaf_instance_masks_panoptic import (
    evaluate_leaf_instances,
)
from phenobench.evaluation.evaluate_plant_bounding_boxes import evaluate_plant_detection
from phenobench.evaluation.evaluate_plant_instance_masks_panoptic import (
    evaluate_plant_instances,
)
from phenobench.evaluation.evaluate_semantics import evaluate_semantics


def run():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--task',
      choices=['semantics', 'panoptic', 'leaf_instances', 'plant_detection', 'leaf_detection', 'hierarchical'],
      required=True,
      type=str,
      help='task to evaluate.')
  parser.add_argument('--phenobench_dir', required=True, type=Path, help='Path to the root directory of PhenoBench.')
  parser.add_argument('--prediction_dir', required=True, type=Path, help='Path to prediction directory.')
  parser.add_argument('--split',
                      choices=['train', 'val'],
                      required=True,
                      type=str,
                      default='val',
                      help='Specify which split to use for evaluation.')
  args = vars(parser.parse_known_args()[0])

  if args["task"] == "semantics":
    print(centered_text("Computing metrics for semantic segmentation"))

    results = evaluate_semantics(args)

    print(centered_text(f"Metrics for {args['split']} split of PhenoBench"))

    print(f"{'IoU (crop)':10}: {results['crop']:4}")
    print(f"{'IoU (weed)':10}: {results['weed']:4}")
    print(f"{'mIoU':10}: {results['mIoU']:4}")

  elif args["task"] == "panoptic":
    print(centered_text("Computing metrics for plant panoptic segmentation"))

    semantic_results = evaluate_semantics(args)
    instance_results = evaluate_plant_instances(args)

    print(centered_text(f"Metrics for {args['split']} split of PhenoBench"))

    iou_soil = semantic_results['soil']
    pq_crop = instance_results['plants_cls'][1]['pq']
    pq_weed = instance_results['plants_cls'][2]['pq']

    print(f"{'IoU (soil)':10}: {iou_soil:4}")
    print(f"{'PQ (crop)':10}: {pq_crop:4}")
    print(f"{'PQ (weed)':10}: {pq_weed:4}")
    print(f"{'PQ+':10}: {(iou_soil+pq_crop+pq_weed)/3:4.2f}")

  elif args["task"] == "leaf_instances":
    print(centered_text("Computing metrics for leaf instance segmentation"))

    results = evaluate_leaf_instances(args)

    print(centered_text(f"Metrics for {args['split']} split of PhenoBench"))
    pq_leaf = results["leaves_pq"]
    print(f"{'PQ (leaf)':10}: {pq_leaf:4}")

  elif args["task"] == "plant_detection":
    print(centered_text("Computing metrics for plant detection"))

    results = evaluate_plant_detection(args)

    print(centered_text(f"Metrics for {args['split']} split of PhenoBench"))

    print(f"{'mAP':10}: {results['mAP']:4}")
    print(f"{'mAP_50':10}: {results['mAP_50']:4}")
    print(f"{'mAP_75':10}: {results['mAP_75']:4}")
    print(f"{'AP (crop)':10}: {results['mAP_cls'][0]:4}")
    print(f"{'AP (weed)':10}: {results['mAP_cls'][1]:4}")

  elif args["task"] == "leaf_detection":
    print(centered_text("Computing metrics for leaf detection"))

    results = evaluate_leaf_detection(args)

    print(centered_text(f"Metrics for {args['split']} split of PhenoBench"))

    print(f"{'mAP':10}: {results['mAP']:4}")
    print(f"{'mAP_50':10}: {results['mAP_50']:4}")
    print(f"{'mAP_75':10}: {results['mAP_75']:4}")

  elif args["task"] == "hierarchical":
    print(centered_text("Computing metrics for leaf detection"))

    semantic_results = evaluate_semantics(args)
    instance_results = evaluate_plant_instances(args)
    leaf_results = evaluate_leaf_instances(args)

    print(centered_text(f"Metrics for {args['split']} split of PhenoBench"))

    iou_soil = semantic_results['soil']
    iou_weed = semantic_results['weed']

    pq_leaf = leaf_results["leaves_pq"]
    pq_crop = instance_results['plants_cls'][1]['pq']

    print(f"{'IoU (soil)':10}: {iou_soil:4}")
    print(f"{'IoU (weed)':10}: {iou_weed:4}")
    print(f"{'PQ (leaf)':10}: {pq_leaf:4}")
    print(f"{'PQ (crop)':10}: {pq_crop:4}")
    print(f"{'PQ':10}: {(pq_crop+pq_leaf)/2:4.2f}")
    print(f"{'PQ+':10}: {(iou_soil+iou_weed+pq_crop+pq_leaf)/4:4.2f}")


if __name__ == '__main__':
  run()