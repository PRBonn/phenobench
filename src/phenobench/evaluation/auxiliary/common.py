import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from numpy import genfromtxt
from PIL import Image


def is_png(filename: str) -> bool:
  return filename.endswith('.png')


def is_txt(filename: str) -> bool:
  return filename.endswith('.txt')


def get_png_files_in_dir(path_to_dir: Path) -> List[str]:
  filenames = sorted([filename for filename in os.listdir(path_to_dir) if is_png(filename)])

  return filenames


def get_txt_files_in_dir(path_to_dir: Path) -> List[str]:
  filenames = sorted([filename for filename in os.listdir(path_to_dir) if is_txt(filename)])

  return filenames

def load_file_as_int_tensor(path_to_file: Path) -> torch.Tensor:

  tensor = Image.open(path_to_file)
  tensor = np.asarray(tensor).astype(np.int32)
  tensor = torch.Tensor(tensor).type(torch.int32)

  return tensor

def load_file_as_tensor(path_to_file: Path) -> torch.Tensor:
  to_tensor = T.ToTensor()

  tensor = Image.open(path_to_file)
  tensor = to_tensor(tensor)

  return tensor


def load_yolo_txt_file(path_to_file: Path) -> torch.Tensor:
  annos = genfromtxt(path_to_file)

  return torch.Tensor(annos)

def centered_text(text: str, line_width: int = 80, fill: str = '=') -> str:
  """ generate string such that text is center in given line width using the fill """
  fill_width = (line_width - len(text)-2)//2
  repeat = fill_width // len(fill)

  return f"{fill*repeat} {text} {fill*repeat}"
