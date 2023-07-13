import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import yaml
from torchmetrics.classification import MulticlassJaccardIndex  # type: ignore
from tqdm import tqdm

from phenobench.evaluation.auxiliary.common import (
    get_png_files_in_dir,
    load_file_as_int_tensor,
    load_file_as_tensor,
)
from phenobench.evaluation.auxiliary.convert import convert_partial_semantics

torch.set_printoptions(precision=4, sci_mode=False)


def parse_args() -> Dict:
  parser = argparse.ArgumentParser()
  parser.add_argument('--phenobench_dir', required=True, type=Path, help='Path to the root directory of PhenoBench.')
  parser.add_argument('--prediction_dir', required=True, type=Path, help='Path to prediction directory.')
  parser.add_argument('--export', required=False, type=Path, help='Path to export directory.')
  parser.add_argument('--split',
                      choices=['train', 'val', 'test'],
                      required=True,
                      type=str,
                      help='Specify which split to use for evaluation.')
  parser.add_argument('--mapping', required=False, type=Path, help='Path to mapping.json file (only for test-split)')

  # Required directory structure
  # ├── PhenoBench
  # |   └── <split>
  # │       └── semantics
  # └── prediction
  #     └── semantics

  args = vars(parser.parse_known_args()[0])
  if "export" in args: args['export'].mkdir(parents=True, exist_ok=True)

  if args['split'] == 'test':
    if (not args['mapping']):
      raise TypeError(f"For the evaluation on the test-split you need to specify the '--mapping' arguments.")

  return args


def evaluate_semantics(args: Dict[str, str]) -> Dict[str, float]:
  """ Compute semantic segmentation scores: IoU for "crop", "weed", "soil" and "mIoU".

  Args:
      args (Dict): command line arguments specifying the required paths

  Returns:
      Dict: metrics for classes "soil", "crop", "weed"
  """

  # ------- Get all ground truth and prediction files -------
  gt_fnames = get_png_files_in_dir(args['phenobench_dir'] / args['split'] / 'semantics')
  pred_fnames = get_png_files_in_dir(args['prediction_dir'] / 'semantics')

  # Load the mapping from randomized to original fnames
  if args['split'] == 'test':
    with open(args['mapping'], 'r') as istream:
      fname_mapping = json.load(istream)
    fname_mapping_reverse = {val: key for key, val in fname_mapping.items()}

    pred_original_fnames = []
    for pred_fname in pred_fnames:
      pred_original_fnames.append(fname_mapping[pred_fname])
    gt_fnames.sort()
    pred_original_fnames.sort()
  else:
    pred_original_fnames = pred_fnames
    fname_mapping_reverse = {fname: fname for fname in pred_fnames}

  # ------- Setup evaluator -------
  evaluator = MulticlassJaccardIndex(num_classes=3, average=None)
  if args['split'] == 'test':
    evaluator_2020 = MulticlassJaccardIndex(num_classes=3, average=None)
    evaluator_2021 = MulticlassJaccardIndex(num_classes=3, average=None)

  n_total = len(gt_fnames)
  for gt_fname, pred_fname in tqdm(zip(gt_fnames, pred_original_fnames), total=n_total):
    assert gt_fname == pred_fname

    semantics_gt = convert_partial_semantics(
        load_file_as_tensor(args['phenobench_dir'] / args['split'] / 'semantics' / gt_fname)).squeeze()
    semantics_pred = convert_partial_semantics(
        load_file_as_int_tensor(args['prediction_dir'] / 'semantics' / fname_mapping_reverse[pred_fname])).squeeze()

    if args['split'] == 'test':
      if '2021' not in gt_fname:
        evaluator_2020.update(semantics_pred, semantics_gt)
      if '2021' in gt_fname:
        evaluator_2021.update(semantics_pred, semantics_gt)

    evaluator.update(semantics_pred, semantics_gt)

  metrics = evaluator.compute() * 100.0  # tensor of shape [3]

  eval_results = {}
  if args['split'] == 'test':
    ext = "_all"
  else:
    ext = ""
  eval_results[f"soil{ext}"] = round(float(metrics[0]), 2)
  eval_results[f"crop{ext}"] = round(float(metrics[1]), 2)
  eval_results[f"weed{ext}"] = round(float(metrics[2]), 2)
  eval_results[f"mIoU{ext}"] = round(float(metrics.mean()), 2)

  if "export" in args:
    fpath_out_all = args['export'] / 'all'
    fpath_out_all.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_all / 'eval_semantics.yaml', 'w') as ostream:
      yaml.dump(eval_results, ostream)

  if args['split'] == 'test':
    # 2020
    eval_results_2020 = {}
    metrics_2020 = evaluator_2020.compute() * 100.0
    ext = "_2020"
    eval_results_2020[f"soil{ext}"] = round(float(metrics_2020[0]), 2)
    eval_results_2020[f"crop{ext}"] = round(float(metrics_2020[1]), 2)
    eval_results_2020[f"weed{ext}"] = round(float(metrics_2020[2]), 2)
    eval_results_2020[f"mIoU{ext}"] = round(float(metrics_2020.mean()), 2)

    fpath_out_2020 = args['export'] / '2020'
    fpath_out_2020.mkdir(parents=True, exist_ok=True)
    with open(fpath_out_2020 / 'eval_semantics.yaml', 'w') as ostream:
      yaml.dump(eval_results_2020, ostream)

    # 2021
    eval_results_2021 = {}
    metrics_2021 = evaluator_2021.compute() * 100.0
    ext = "_2021"
    eval_results_2021[f"soil{ext}"] = round(float(metrics_2021[0]), 2)
    eval_results_2021[f"crop{ext}"] = round(float(metrics_2021[1]), 2)
    eval_results_2021[f"weed{ext}"] = round(float(metrics_2021[2]), 2)
    eval_results_2021[f"mIoU{ext}"] = round(float(metrics_2021.mean()), 2)

    if "export" in args:
      fpath_out_2021 = args['export'] / '2021'
      fpath_out_2021.mkdir(parents=True, exist_ok=True)
      with open(fpath_out_2021 / 'eval_semantics.yaml', 'w') as ostream:
        yaml.dump(eval_results_2021, ostream)

  return eval_results


if __name__ == '__main__':
  args = parse_args()
  evaluate_semantics(args)
