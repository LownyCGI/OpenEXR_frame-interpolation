# Copyright 2022 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for frame interpolation on a set of video frames."""
import os
import shutil
from typing import Generator, Iterable, List, Optional

from . import interpolator as interpolator_lib
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import OpenEXR
import Imath

_UINT8_MAX_F = float(np.iinfo(np.uint8).max)
_CONFIG_FFMPEG_NAME_OR_PATH = 'ffmpeg'


def read_image(filename: str) -> np.ndarray:
    """Reads an 32-bit EXR image and returns the RGB channels.
    Args:
        filename: The input filename to read.
    Returns:
        A float32 3-channel (RGB) ndarray with colors in the [0..1] range.
    """
    # Read EXR file
    exr_file = OpenEXR.InputFile(filename)
    header = exr_file.header()
    size = (header['dataWindow'].max.x + 1, header['dataWindow'].max.y + 1)

    # Read only RGB channels
    r_channel = np.frombuffer(exr_file.channel('R'), dtype=np.float32)
    g_channel = np.frombuffer(exr_file.channel('G'), dtype=np.float32)
    b_channel = np.frombuffer(exr_file.channel('B'), dtype=np.float32)

    # Combine channels and reshape to image size
    image = np.stack([r_channel, g_channel, b_channel], axis=-1).reshape(size[1], size[0], -1)

    
    return image


def write_image(filename: str, image: np.ndarray) -> None:
    """Writes a float16 3-channel RGB ndarray image, with colors in range [0..1].

    Args:
        filename: The output filename to save.
        image: A float16 3-channel (RGB) ndarray with colors in the [0..1] range.
    """
    # Convert to half-precision (float16)
    image = image.astype(np.float16)

    # Create EXR file and write RGB channels
    header = OpenEXR.Header(image.shape[1], image.shape[0])
    header['channels'] = {'R': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)), 
                          'G': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF)), 
                          'B': Imath.Channel(Imath.PixelType(Imath.PixelType.HALF))}
    output = OpenEXR.OutputFile(filename, header)
    output.writePixels({'R': image[..., 0].tobytes(), 
                        'G': image[..., 1].tobytes(), 
                        'B': image[..., 2].tobytes()})


def _recursive_generator(
    frame1: np.ndarray, frame2: np.ndarray, num_recursions: int,
    interpolator: interpolator_lib.Interpolator,
    bar: Optional[tqdm] = None
) -> Generator[np.ndarray, None, None]:
  """Splits halfway to repeatedly generate more frames.

  Args:
    frame1: Input image 1.
    frame2: Input image 2.
    num_recursions: How many times to interpolate the consecutive image pairs.
    interpolator: The frame interpolator instance.

  Yields:
    The interpolated frames, including the first frame (frame1), but excluding
    the final frame2.
  """
  if num_recursions == 0:
    yield frame1
  else:
    # Adds the batch dimension to all inputs before calling the interpolator,
    # and remove it afterwards.
    time = np.full(shape=(1,), fill_value=0.5, dtype=np.float32)
    mid_frame = interpolator(frame1[np.newaxis, ...], frame2[np.newaxis, ...],
                             time)[0]
    bar.update(1) if bar is not None else bar
    yield from _recursive_generator(frame1, mid_frame, num_recursions - 1,
                                    interpolator, bar)
    yield from _recursive_generator(mid_frame, frame2, num_recursions - 1,
                                    interpolator, bar)


def interpolate_recursively_from_files(
    frames: List[str], times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  Loads the files on demand and uses the yield paradigm to return the frames
  to allow streamed processing of longer videos.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  num_frames = (n - 1) * (2**(times_to_interpolate) - 1)
  bar = tqdm(total=num_frames, ncols=100, colour='green')
  for i in range(1, n):
    yield from _recursive_generator(
        read_image(frames[i - 1]), read_image(frames[i]), times_to_interpolate,
        interpolator, bar)
  # Separately yield the final frame.
  yield read_image(frames[-1])

def interpolate_recursively_from_memory(
    frames: List[np.ndarray], times_to_interpolate: int,
    interpolator: interpolator_lib.Interpolator) -> Iterable[np.ndarray]:
  """Generates interpolated frames by repeatedly interpolating the midpoint.

  This is functionally equivalent to interpolate_recursively_from_files(), but
  expects the inputs frames in memory, instead of loading them on demand.

  Recursive interpolation is useful if the interpolator is trained to predict
  frames at midpoint only and is thus expected to perform poorly elsewhere.

  Args:
    frames: List of input frames. Expected shape (H, W, 3). The colors should be
      in the range[0, 1] and in gamma space.
    times_to_interpolate: Number of times to do recursive midpoint
      interpolation.
    interpolator: The frame interpolation model to use.

  Yields:
    The interpolated frames (including the inputs).
  """
  n = len(frames)
  num_frames = (n - 1) * (2**(times_to_interpolate) - 1)
  bar = tqdm(total=num_frames, ncols=100, colour='green')
  for i in range(1, n):
    yield from _recursive_generator(frames[i - 1], frames[i],
                                    times_to_interpolate, interpolator, bar)
  # Separately yield the final frame.
  yield frames[-1]


def get_ffmpeg_path() -> str:
  path = shutil.which(_CONFIG_FFMPEG_NAME_OR_PATH)
  if not path:
    raise RuntimeError(
        f"Program '{_CONFIG_FFMPEG_NAME_OR_PATH}' is not found;"
        " perhaps install ffmpeg using 'apt-get install ffmpeg'.")
  return path
