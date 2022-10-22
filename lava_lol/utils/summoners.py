import cv2
import numpy as np

from utils.constants import Borders

def get_summoner_spells(self, cap):
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    borders = Borders()

    roles = 

    match_summoners = {
        'red': {k: [] for k in roles},
        'blue': {k: [] for k in roles}
    }

    for side in ['red', 'blue']:
        for role in roles:
            for i in range(2):
                cropped = gray[
                    constants.summoner_spells[role][i],
                    constants.summoner_spells[side]]
                threshold = 0.

                for summoner in constants.summoners.keys():
                    max_probability = np.max(
                        cv2.matchTemplate(
                            cropped,
                            constants.summoners[summoner],
                            cv2.TM_CCOEFF_NORMED)
                    )

                    if max_probability > threshold:
                        summoner_temp = summoner
                        threshold = max_probability

                match_summoners[side][role].append(summoner_temp)

    return match_summoners