import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # type: ignore
from tqdm import tqdm

from phenobench.evaluation.auxiliary.common import (
    get_png_files_in_dir,
    get_txt_files_in_dir,
    load_file_as_tensor,
    load_yolo_txt_file,
)
from phenobench.evaluation.auxiliary.convert import (
    cvt_gt_to_bbox_map,
    cvt_yolo_to_bbox_map,
)
from phenobench.evaluation.auxiliary.filter import filter_partials_boxes


def parse_args() -> Dict:
  parser = argparse.ArgumentParser()
  parser.add_argument('--phenobench_dir', required=True, type=Path, help='Path to ground truth directory.')
  parser.add_argument('--prediction_dir', required=True, type=Path, help='Path to prediction directory.')
  parser.add_argument('--export', required=True, type=Path, help='Path to export directory.')
  parser.add_argument('--split', choices=['train', 'val', 'test'], required=True, type=str, help='Specify which split to use for evaluation.')
  
  parser.add_argument('--mapping', required=False, type=Path, help='Path to mapping.json file (only for test-split)')
  # Required directory structure
  # ├── ground-truth
  # │   ├── leaf_instances
  # │   ├── leaf_visibility
  # └── prediction
  #     └── leaf_bboxes

  args = vars(parser.parse_args())

  args['export'].mkdir(parents=True, exist_ok=True)

  if args['split'] == 'test':
    if (not args['mapping']):
      raise TypeError(f"For the evaluation on the test-split you need to specify the '--date' and '--mapping' arguments.")

  return args


def evaluate_leaf_detection(args:Dict[str,str]) -> Dict[str,str]:

  # ------- Ground Truth -------
  gt_instance_fnames = get_png_files_in_dir(args['phenobench_dir']  / args["split"] / 'leaf_instances')
  gt_visibility_fnames = get_png_files_in_dir(args['phenobench_dir']  / args["split"] / 'leaf_visibility')

  # ------- Predictions -------
  pred_bboxes_fnames = get_txt_files_in_dir(args['prediction_dir'] / 'leaf_bboxes')

  # ------- Load the mapping from randomized to original fnames -------
  if args['split'] == 'test':
    with open(args['mapping'], 'r') as istream:
      fname_mapping = json.load(istream)
    fname_mapping_reverse = {val: key for key, val in fname_mapping.items()}

    pred_bboxes_original_fnames = []
    for pred_fname in pred_bboxes_fnames:
      pred_bboxes_original_fnames.append(fname_mapping[pred_fname])

    gt_instance_fnames.sort()
    gt_visibility_fnames.sort()
    pred_bboxes_original_fnames.sort()
  else:
    pred_bboxes_original_fnames = pred_bboxes_fnames
    fname_mapping_reverse = {fname:fname for fname in pred_bboxes_fnames}

  # ------- Setup evaluator -------
  evaluator = MeanAveragePrecision(box_format='cxcywh')
  if args['split'] == 'test':
    evaluator_2020 = MeanAveragePrecision(box_format='cxcywh')
    evaluator_2021 = MeanAveragePrecision(box_format='cxcywh')

  for gt_instance_fname, gt_visibility_fname, pred_bboxes_fnames in tqdm(zip(gt_instance_fnames, gt_visibility_fnames, pred_bboxes_original_fnames),total=len(gt_instance_fnames)):
    assert gt_instance_fname.split('.')[0] == gt_visibility_fname.split('.')[0] == pred_bboxes_fnames.split('.')[0]

    gt_instance_map = load_file_as_tensor(args['phenobench_dir']  / args["split"] / 'leaf_instances' / gt_instance_fname).squeeze()  # [H x W]
    gt_visibility = load_file_as_tensor(args['phenobench_dir']  / args["split"] / 'leaf_visibility' / gt_visibility_fname).squeeze()  # [H x W]
    gt_semantics = (gt_instance_map > 0).type(torch.uint8)  # derive semantics from instance map
    targets = cvt_gt_to_bbox_map(gt_instance_map, gt_semantics, gt_visibility)

    yolo_pred = load_yolo_txt_file(args['prediction_dir'] / 'leaf_bboxes' / fname_mapping_reverse[pred_bboxes_fnames])
    preds = cvt_yolo_to_bbox_map(yolo_pred)

    filter_partials_boxes(preds[0], targets[0])

    if args['split'] == 'test':
      if '2021' not in gt_instance_fname:
        evaluator_2020.update(preds, targets)
      if '2021' in gt_instance_fname:
        evaluator_2021.update(preds, targets)

    evaluator.update(preds, targets)

  metrics = evaluator.compute()
  eval_results = {}
  if args['split'] == 'test':
    ext = "_all"
  else:
    ext = ""
  eval_results[f'mAP{ext}'] = round(float(metrics['map'] * 100),2)
  eval_results[f'mAP_50{ext}'] = round(float(metrics['map_50'] * 100), 2)
  eval_results[f'mAP_75{ext}'] = round(float(metrics['map_75'] * 100), 2)
  
  if "export" in args:
    fpath_out_all = args['export'] / 'all'
    fpath_out_all.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_all / 'eval_leaf_bboxes.yaml', 'w') as ostream:
      yaml.dump(eval_results, ostream)

  if args['split'] == 'test':
    metrics_2020 = evaluator_2020.compute()
    eval_results_2020 = {}
    ext = "_2020"
    eval_results_2020[f'mAP{ext}'] = round(float(metrics_2020['map'] * 100), 2)
    eval_results_2020[f'mAP_50{ext}'] = round(float(metrics_2020['map_50'] * 100), 2)
    eval_results_2020[f'mAP_75{ext}'] = round(float(metrics_2020['map_75'] * 100), 2)
    
    fpath_out_2020 = args['export'] / '2020'
    fpath_out_2020.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_2020 / 'eval_leaf_bboxes.yaml', 'w') as ostream:
      yaml.dump(eval_results_2020, ostream)

    metrics_2021 = evaluator_2021.compute()
    eval_results_2021 = {}
    ext = "_2021"
    eval_results_2021[f'mAP{ext}'] = round(float(metrics_2021['map'] * 100), 2)
    eval_results_2021[f'mAP_50{ext}'] = round(float(metrics_2021['map_50'] * 100), 2)
    eval_results_2021[f'mAP_75{ext}'] = round(float(metrics_2021['map_75'] * 100), 2)
    
    if "export" in args:
      fpath_out_2021 = args['export'] / '2021'
      fpath_out_2021.mkdir(parents=True, exist_ok=True)
      with open(fpath_out_2021 / 'eval_leaf_bboxes.yaml', 'w') as ostream:
        yaml.dump(eval_results_2021, ostream)

  return eval_results

if __name__ == '__main__':
  args = parse_args()
  evaluate_leaf_detection(args)
