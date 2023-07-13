#!/usr/bin/env python3

import argparse
import os
from zipfile import ZipFile

import numpy as np


class ValidationException(Exception):
  pass


def run():
  parser = argparse.ArgumentParser(
      description=
      "Validate a submission zip file needed to evaluate on CodaLab competitions of PhenoBench.\n\nThe validator checks:\n  1. correct folder structure,\n  2. existence of label files for each image.\nInvalid labels are ignored by the evaluation script, therefore we don't check\nfor invalid labels.",
      formatter_class=argparse.RawTextHelpFormatter)

  parser.add_argument(
      "--task",
      type=str,
      required=True,
      choices=["semantics", "panoptic", "leaf_instances", "plant_detection", "leaf_detection", "hierarchical"],
      help='task for which the zip file should be validated.')

  parser.add_argument(
      "--zipfile",
      type=str,
      required=True,
      help='zip file that should be validated.',
  )

  parser.add_argument('--phenobench_dir',
                      type=str,
                      required=True,
                      help='Root directory containing the folder train, val, test of the PhenoBench.')

  FLAGS, _ = parser.parse_known_args()

  checkmark = "\u2713"

  required_folders = {
      "semantics": ["semantics"],
      "panoptic": ["semantics", "plant_instances"],
      "leaf_instances": ["leaf_instances"],
      "plant_detection": ["plant_bboxes"],
      "leaf_detection": ["leaf_bboxes"],
      "hierarchical": ["semantics", "plant_instances", "leaf_instances"]
  }

  try:

    print('Validating zip archive "{}".\n'.format(FLAGS.zipfile))

    print(" ========== {:^15} ========== ".format(FLAGS.task))

    print("  1. Checking filename.............. ", end="", flush=True)
    if not FLAGS.zipfile.endswith('.zip'):
      raise ValidationException('Competition submission must end with ".zip"')
    print(checkmark)

    with ZipFile(FLAGS.zipfile) as zipfile:

      print("  2. Checking directory structure... ", end="", flush=True)

      directories = [folder.filename for folder in zipfile.infolist() if folder.filename.endswith("/")]
      for expected_folder in required_folders[FLAGS.task]:
        if expected_folder + "/" not in directories:
          raise ValidationException(
              f'Directory "{expected_folder}" missing inside zip file. Your zip file should contain following folders: {", ".join(required_folders[FLAGS.task])}'
          )

      print(checkmark)

      print('  3. Checking files................. ', end='', flush=True)

      prediction_files = {info.filename: info for info in zipfile.infolist() if not info.filename.endswith("/")}
      image_files = sorted(os.listdir(os.path.join(FLAGS.phenobench_dir, "test", "images")))

      for expected_folder in required_folders[FLAGS.task]:
        if "bboxes" in expected_folder:
          for image_file in image_files:
            expected_file = f"{expected_folder}/{os.path.splitext(image_file)[0]}.txt"
            if expected_file not in prediction_files:
              raise ValidationException(f'Missing prediction for {image_file} in folder {expected_folder}!')
        else:
          for image_file in image_files:
            expected_file = f"{expected_folder}/{image_file}"
            if expected_file not in prediction_files:
              raise ValidationException(f'Missing prediction for {image_file} in folder {expected_folder}!')

      print(checkmark)

  except ValidationException as ex:
    print("\n\n  " + "\u001b[1;31m>>> Error: " + str(ex) + "\u001b[0m")
    exit(1)

  print("\n\u001b[1;32mEverything ready for submission!\u001b[0m  \U0001F389")


if __name__ == "__main__":
  run()
