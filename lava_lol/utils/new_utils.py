from typing import Dict

import cv2

from .video_file_helper import parse_local_files
from .youtube_helper import parse_youtube_url, is_valid_youtube_url


def parse_url(url: str) -> Dict[str, str]:
    """
    """
    if is_valid_youtube_url(url):
        return parse_youtube_url(url)
    else:
        return parse_local_files(url)


def reset_capture(cap: cv2.VideoCapture,
                  frames_to_skip: int,
                  count: int) -> cv2.VideoCapture:
    """
    """
    cap.set(1, frames_to_skip * 120 * (count - 2))

    return cap