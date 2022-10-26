from typing import Dict, List

import cv2
import numpy as np

from ..utils.constants import Borders
from ..utils.overlay_utils import get_summoner_spells


def identify_summoner_spells(cap: cv2.VideoCapture,
                             frames_to_skip: int,
                             summoner_threshold: float = 0.7
                             ) -> Dict[str, Dict[str, List[List[str]]]]:
    """
    """
    borders = Borders()

    roles = Borders.ROLES

    match_summoners = {
        'blue': {k: [['', summoner_threshold], ['', summoner_threshold]] for k in roles},
        'red': {k: [['', summoner_threshold], ['', summoner_threshold]] for k in roles}
    }

    summoners = get_summoner_spells()

    ret, frame = cap.read()

    while ret:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        for side in match_summoners:
            for role in match_summoners[side]:
                for i in range(2):
                    summoner_temp = ""
                    threshold = match_summoners[side][role][i][1]

                    cropped = gray[
                        borders.SUMMONER_SPELL_BORDERS[role][i],
                        borders.SUMMONER_SPELL_BORDERS[side]
                    ]

                    for summoner in summoners.keys():
                        max_probability = np.max(
                            cv2.matchTemplate(
                                cropped,
                                summoners[summoner],
                                cv2.TM_CCOEFF_NORMED)
                        )

                        if max_probability > threshold:
                            summoner_temp = summoner
                            threshold = max_probability

                    if summoner_temp != '':
                        match_summoners[side][role][i][0] = summoner_temp
                        match_summoners[side][role][i][1] = threshold

        for side in match_summoners.values():
            for role in side.values():
                if role[0][0] == '' or role[1][0] == '':
                    for _ in range(frames_to_skip):
                        cap.grab()

                    ret, frame = cap.read()
                    continue
                ret = False

    match_summoners =  {
        side: {
            role: [
                player[0][0].split('/')[-1],
                player[1][0].split('/')[-1]
            ] for role, player in roles.items()
        } for side, roles in match_summoners.items()
    }

    return match_summoners
