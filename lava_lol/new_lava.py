from typing import List, Dict

import cv2

from utils.logger import make_logger
from utils.new_utils import parse_url
from utils.headers import get_headers
from utils.identify_champions import identify_champions

class LAVA():
    """
    """
    def __init__(self):
        pass

    def gather_data(self, url: str, show_minimap: bool = True) -> None:
        """
        """
        videos = parse_url(url)

        logger = make_logger('lava')

        for video, url in videos.items():
            cap = cv2.VideoCapture(url)
            frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS))

            count, header, overlay = get_headers(cap, frames_to_skip)

            # get_headers found the start of the match, skip to it
            cap.set(1, frames_to_skip * 120 * (count - 2))

            champs = identify_champions(cap, header, frames_to_skip)

            summoners = get_summoner_spells(cap, frames_to_skip)

    def parse_champions(self, url: str) -> List[Dict[str, str]]:
        """
        """
        videos = parse_url(url)

        champ_info = []

        for video, url in videos.items():
            cap = cv2.VideoCapture(url)



    def draw_graphs(self):
        """
        """
        pass
