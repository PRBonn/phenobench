import argparse
import json
from pathlib import Path
from typing import Dict

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
  parser.add_argument('--export', required=False, type=Path, help='Path to export directory.')
  parser.add_argument('--split',
                      choices=['train', 'val', 'test'],
                      required=True,
                      type=str,
                      help='Specify which split to use for evaluation.')
  parser.add_argument('--mapping', required=False, type=Path, help='Path to mapping.json file (only for test-split)')

  # Required directory structure
  # ├── ground-truth
  # |   └── <split>
  # │       ├── plant_instances
  # |       ├── plant_visibility
  # │       └── semantics
  # └── prediction
  #     └── plant_instances
  #     └── semantics

  args = vars(parser.parse_args())
  args['export'].mkdir(parents=True, exist_ok=True)

  if args['split'] == 'test':
    if (not args['mapping']):
      raise TypeError(
          f"For the evaluation on the test-split you need to specify the '--date' and '--mapping' arguments.")

  return args


def evaluate_plant_instances(args: Dict[str, str]) -> Dict[str, float]:

  # ------- Ground Truth -------
  gt_instance_fnames = get_png_files_in_dir(args['phenobench_dir'] / args['split'] / 'plant_instances')
  gt_semantic_fnames = get_png_files_in_dir(args['phenobench_dir'] / args['split'] / 'semantics')
  gt_visibility_fnames = get_png_files_in_dir(args['phenobench_dir'] / args['split'] / 'plant_visibility')

  # ------- Predictions -------
  pred_instance_fnames = get_png_files_in_dir(args['prediction_dir'] / 'plant_instances')
  pred_semantic_fnames = get_png_files_in_dir(args['prediction_dir'] / 'semantics')

  # ------- Load the mapping from randomized to original fnames -------
  if args['split'] == 'test':
    with open(args['mapping'], 'r') as istream:
      fname_mapping = json.load(istream)
    fname_mapping_reverse = {val: key for key, val in fname_mapping.items()}

    pred_instance_original_fnames = []
    pred_semantic_original_fnames = []
    for pred_inst_fname, pred_sem_fname in zip(pred_instance_fnames, pred_semantic_fnames):
      pred_instance_original_fnames.append(fname_mapping[pred_inst_fname])
      pred_semantic_original_fnames.append(fname_mapping[pred_sem_fname])
    gt_instance_fnames.sort()
    gt_semantic_fnames.sort()
    gt_visibility_fnames.sort()
    pred_instance_original_fnames.sort()
    pred_semantic_original_fnames.sort()
  else:
    pred_instance_original_fnames = pred_instance_fnames
    pred_semantic_original_fnames = pred_semantic_fnames
    fname_mapping_reverse = {fname: fname for fname in pred_instance_fnames}

  # ------- Setup evaluator -------
  evaluator = PanopticEvaluator()
  if args['split'] == 'test':
    evaluator_2020 = PanopticEvaluator()
    evaluator_2021 = PanopticEvaluator()

  n_total = len(gt_instance_fnames)
  for gt_instance_fname, gt_semantic_fname, gt_visibility_fname, pred_instance_fname, pred_semantic_fname, in tqdm(
      zip(gt_instance_fnames, gt_semantic_fnames, gt_visibility_fnames, pred_instance_original_fnames,
          pred_semantic_original_fnames),
      total=n_total):
    assert gt_instance_fname == gt_semantic_fname == gt_visibility_fname == pred_instance_fname == pred_semantic_fname

    gt_instance_map = load_file_as_tensor(args['phenobench_dir'] / args['split'] / 'plant_instances' /
                                          gt_instance_fname).squeeze()  # [H x W]
    gt_semantics = load_file_as_tensor(args['phenobench_dir'] / args['split'] / 'semantics' /
                                       gt_semantic_fname).squeeze()  # [H x W]
    gt_visibility = load_file_as_tensor(args['phenobench_dir'] / args['split'] / 'plant_visibility' /
                                        gt_instance_fname).squeeze()  # [H x W]

    pred_instance_map = load_file_as_int_tensor(args['prediction_dir'] / 'plant_instances' /
                                                fname_mapping_reverse[pred_instance_fname]).squeeze()  # [H x W]
    pred_semantics = load_file_as_int_tensor(args['prediction_dir'] / 'semantics' /
                                             fname_mapping_reverse[pred_semantic_fname]).squeeze()  # [H x W]

    filter_partial_masks(pred_instance_map, pred_semantics, gt_instance_map, gt_semantics, gt_visibility)
    if args['split'] == 'test':
      if '2021' not in gt_instance_fname:
        evaluator_2020.update(pred_semantics, gt_semantics, pred_instance_map, gt_instance_map)
      if '2021' in gt_instance_fname:
        evaluator_2021.update(pred_semantics, gt_semantics, pred_instance_map, gt_instance_map)

    evaluator.update(pred_semantics, gt_semantics, pred_instance_map, gt_instance_map)

  metrics = round(float(evaluator.compute() * 100), 2)
  metrics_classwise = evaluator.results_classwise
  for class_id in metrics_classwise:
    metrics_classwise[class_id]['pq'] = round(float(metrics_classwise[class_id]['pq'] * 100), 2)

  eval_results = {}
  if args['split'] == 'test':
    ext = "_all"
  else:
    ext = ""

  eval_results[f"plants_pq{ext}"] = metrics
  eval_results[f"plants_cls{ext}"] = metrics_classwise

  if "export" in args:
    fpath_out_all = args['export'] / 'all'
    fpath_out_all.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_all / 'eval_plant_pq.yaml', 'w') as ostream:
      yaml.dump(eval_results, ostream)

  if args['split'] == 'test':
    # 2020
    metrics_2020 = round(float(evaluator_2020.compute() * 100), 2)
    metrics_2020_classwise = evaluator_2020.results_classwise
    for class_id in metrics_2020_classwise:
      metrics_2020_classwise[class_id]['pq'] = round(float(metrics_2020_classwise[class_id]['pq'] * 100), 2)

    eval_results_2020 = {}
    ext = "_2020"
    eval_results_2020[f"plants_pq{ext}"] = metrics_2020
    eval_results_2020[f"plants_cls{ext}"] = metrics_2020_classwise

    fpath_out_2020 = args['export'] / '2020'
    fpath_out_2020.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_2020 / 'eval_plant_pq.yaml', 'w') as ostream:
      yaml.dump(eval_results_2020, ostream)

    # 2021
    metrics_2021 = round(float(evaluator_2021.compute() * 100), 2)
    metrics_2021_classwise = evaluator_2021.results_classwise
    for class_id in metrics_2021_classwise:
      metrics_2021_classwise[class_id]['pq'] = round(float(metrics_2021_classwise[class_id]['pq'] * 100), 2)

    eval_results_2021 = {}
    ext = "_2021"
    eval_results_2021[f"plants_pq{ext}"] = metrics_2021
    eval_results_2021[f"plants_cls{ext}"] = metrics_2021_classwise
    if "export" in args:
      fpath_out_2021 = args['export'] / '2021'
      fpath_out_2021.mkdir(parents=True, exist_ok=True)
      with open(fpath_out_2021 / 'eval_plant_pq.yaml', 'w') as ostream:
        yaml.dump(eval_results_2021, ostream)

  return eval_results

if __name__ == '__main__':
  args = parse_args()
  evaluate_plant_instances(args)
