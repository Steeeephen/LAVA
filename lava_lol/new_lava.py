from typing import List, Dict

import cv2

from .utils.champions import identify_champions
from .utils.headers import identify_headers
from .utils.logger import make_logger
from .utils.minimap import get_map_borders
from .utils.new_utils import parse_url, reset_capture
from .utils.summoners import identify_summoner_spells
from .utils.tracker import track_champions

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

            count, header, overlay = identify_headers(cap, frames_to_skip)

            champs = identify_champions(
                reset_capture(cap, frames_to_skip, count),
                header,
                frames_to_skip
            )

            # summoners = identify_summoner_spells(
            #     reset_capture(cap, frames_to_skip, count),
            #     frames_to_skip
            # )

            map_borders = get_map_borders(cap, 'lec_summer_2020', frames_to_skip)

            df = track_champions(
                cap,
                frames_to_skip,
                map_borders,
                header,
                champs
            )


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
