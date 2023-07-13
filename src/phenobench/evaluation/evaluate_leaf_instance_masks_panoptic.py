import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from tqdm import tqdm

from phenobench.evaluation.auxiliary.common import (
    get_png_files_in_dir,
    load_file_as_int_tensor,
    load_file_as_tensor,
)
from phenobench.evaluation.auxiliary.filter import filter_partial_masks
from phenobench.evaluation.auxiliary.panoptic_eval import PanopticEvaluator


def parse_args() -> Dict:
  parser = argparse.ArgumentParser()
  parser.add_argument('--phenobench_dir', required=True, type=Path, help='Path to ground truth directory.')
  parser.add_argument('--prediction_dir', required=True, type=Path, help='Path to prediction directory.')
  parser.add_argument('--export', required=True, type=Path, help='Path to export directory.')
  parser.add_argument('--split',
                      choices=['train', 'val', 'test'],
                      required=True,
                      type=str,
                      help='Specify which split to use for evaluation.')

  parser.add_argument('--mapping', required=False, type=Path, help='Path to mapping.json file (only for test-split)')
  parser.add_argument('--limit', required=False, type=int, help="Limit evaluation to the first N number of images.")

  # Required directory structure
  # ├── ground-truth
  # |   ├── leaf_visibility
  # │   └── leaf_instances
  # └── prediction
  #     └── leaf_instances

  args = vars(parser.parse_args())
  args['export'].mkdir(parents=True, exist_ok=True)

  if args['split'] == 'test':
    if (not args['mapping']):
      raise TypeError(f"For the evaluation on the test-split you need to specify the '--mapping' argument.")

  return args


def evaluate_leaf_instances(args) -> Dict[str, float]:

  # ------- Ground Truth -------
  gt_instance_fnames = get_png_files_in_dir(args['phenobench_dir'] / args["split"] / 'leaf_instances')
  gt_visibility_fnames = get_png_files_in_dir(args['phenobench_dir'] / args["split"] / 'leaf_visibility')

  # ------- Prediction -------
  pred_fnames = get_png_files_in_dir(args['prediction_dir'] / 'leaf_instances')

  # ------- Load the mapping from randomized to original fnames -------
  if args['split'] == 'test':
    with open(args['mapping'], 'r') as istream:
      fname_mapping = json.load(istream)
    fname_mapping_reverse = {val: key for key, val in fname_mapping.items()}

    pred_original_fnames = []
    for pred_fname in pred_fnames:
      pred_original_fnames.append(fname_mapping[pred_fname])
    gt_instance_fnames.sort()
    gt_visibility_fnames.sort()
    pred_original_fnames.sort()
  else:
    pred_original_fnames = pred_fnames
    fname_mapping_reverse = {fname: fname for fname in pred_fnames}

  # ------- Setup evaluator -------
  evaluator = PanopticEvaluator()
  if args['split'] == 'test':
    evaluator_2020 = PanopticEvaluator()
    evaluator_2021 = PanopticEvaluator()

  n_total = len(gt_instance_fnames)
  for gt_instance_fname, gt_visibility_fname, pred_fname in tqdm(zip(gt_instance_fnames, gt_visibility_fnames,
                                                                     pred_original_fnames),
                                                                 total=n_total):
    assert gt_instance_fname == gt_visibility_fname == pred_fname

    gt_instance_map = load_file_as_tensor(args['phenobench_dir'] / args["split"] / 'leaf_instances' /
                                          gt_instance_fname).squeeze()  # [H x W]
    gt_visibility = load_file_as_tensor(args['phenobench_dir'] / args["split"] / 'leaf_visibility' /
                                        gt_visibility_fname).squeeze()  # [H x W]
    gt_semantics = (gt_instance_map > 0).type(torch.uint8)  # derive semantics from instance map

    pred_instance_map = load_file_as_int_tensor(args['prediction_dir'] / 'leaf_instances' /
                                                fname_mapping_reverse[pred_fname]).squeeze()  # [H x W]
    pred_semantics = (pred_instance_map > 0).type(torch.uint8)  # derive semantics from instance map

    filter_partial_masks(pred_instance_map, pred_semantics, gt_instance_map, gt_semantics, gt_visibility)

    if args['split'] == 'test':
      if '2021' not in gt_instance_fname:
        evaluator_2020.update(pred_semantics, gt_semantics, pred_instance_map, gt_instance_map)
      if '2021' in gt_instance_fname:
        evaluator_2021.update(pred_semantics, gt_semantics, pred_instance_map, gt_instance_map)

    evaluator.update(pred_semantics, gt_semantics, pred_instance_map, gt_instance_map)

  metrics = round(float(evaluator.compute() * 100), 2)

  eval_results = {}
  if args['split'] == 'test':
    ext = "_all"
  else:
    ext = ""
  eval_results[f'leaves_pq{ext}'] = metrics

  if "export" in args:
    fpath_out_all = args['export'] / 'all'
    fpath_out_all.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_all / 'eval_leaf_pq.yaml', 'w') as ostream:
      yaml.dump(eval_results, ostream)

  if args['split'] == 'test':
    # 2020
    metrics_2020 = round(float(evaluator_2020.compute() * 100), 2)
    eval_results_2020 = {}
    ext = "_2020"
    eval_results_2020[f'leaves_pq{ext}'] = metrics_2020

    fpath_out_2020 = args['export'] / '2020'
    fpath_out_2020.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_2020 / 'eval_leaf_pq.yaml', 'w') as ostream:
      yaml.dump(eval_results_2020, ostream)

    # 2021
    metrics_2021 = round(float(evaluator_2021.compute() * 100), 2)
    eval_results_2021 = {}
    ext = "_2021"
    eval_results_2021[f'leaves_pq{ext}'] = metrics_2021

    fpath_out_2021 = args['export'] / '2021'
    fpath_out_2021.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_2021 / 'eval_leaf_pq.yaml', 'w') as ostream:
      yaml.dump(eval_results_2021, ostream)

  return eval_results


if __name__ == '__main__':
  args = parse_args()
  evaluate_leaf_instances(args)
