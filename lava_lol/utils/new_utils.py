from typing import Dict

from .video_file_helper import parse_local_files
from .youtube_helper import parse_youtube_url, is_valid_youtube_url


def parse_url(url: str) -> Dict[str, str]:
    """
    """
    if is_valid_youtube_url(url):
        return parse_youtube_url(url)
    else:
        return parse_local_files(url)
