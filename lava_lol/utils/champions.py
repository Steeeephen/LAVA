import cv2
import numpy as np

from ..utils.constants import Borders
from ..utils.headers import get_header_borders
from ..utils.overlay_utils import get_assets, get_champion_portraits, get_champion_minimap_icon

def identify_champions(cap: cv2.VideoCapture,
                       header: np.array,
                       frames_to_skip: int,
                       portrait_threshold: float = 0.65,
                       header_threshold: float = 0.75):
    """
    """
    borders = Borders()
    roles = borders.ROLES

    blue_champion_portraits = get_champion_portraits('blue')
    red_champion_portraits = get_champion_portraits('red')

    # Instantiate dictionary for champions
    champions = {
        'blue': {},
        'red': {}
    }

    for role in roles:
        champions['blue'][role] = {'champion': "", 'template': ""}
        champions['red'][role] = {'champion': "", 'template': ""}

    ret, frame = cap.read()
    
    # Grab portraits for identifying the champions played
    identified = 0

    champions_found = []

    # Identify blue side champions until exactly 5 have been found
    while identified != 5:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        identified = 0
        
        # Check the sidebar for each champion
        for role in roles:
            threshold = portrait_threshold
            most_likely_champion = ""

            sidebar_borders = borders.BLUE_PORTRAIT_BORDERS[role]

            # Crop to each role's portrait in the video
            cropped = gray[sidebar_borders]

            # Scroll through each template and find the best match
            for champion_name, champion_portrait in blue_champion_portraits.items():
                if champion_name in champions_found:
                    continue

                champion_classify_percent = np.max(
                    cv2.matchTemplate(
                        cropped,
                        champion_portrait,
                        cv2.TM_CCOEFF_NORMED
                    )
                )

                # If a better match than the previous best, log that champion
                if champion_classify_percent > threshold:
                    threshold = champion_classify_percent
                    most_likely_champion = champion_name
                    champions_found.append(champion_name)

            champions['blue'][role]['champion'] = most_likely_champion
            if most_likely_champion != "":
                identified += 1

                print(f"Blue {role} identified ({100 * threshold:.2f}%): {most_likely_champion}")

        # If too few champions found, skip a frame and try again
        if identified < 5:
            print(f"-- Not enough champions found, trying again --")
            champions_found = []
            for _ in range(frames_to_skip):
                cap.grab()

            ret, frame = cap.read()

    # Same for red side champions
    while identified != 10:
        identified = 5

        for role in roles:
            threshold = portrait_threshold
            most_likely_champion = ""

            sidebar_borders = borders.RED_PORTRAIT_BORDERS[role]

            # Crop to each role's portrait in the video
            cropped = gray[sidebar_borders]

            # Scroll through each template and find the best match
            for champion_name, champion_portrait in red_champion_portraits.items():
                champion_classify_percent = np.max(
                    cv2.matchTemplate(
                        cropped,
                        champion_portrait,
                        cv2.TM_CCOEFF_NORMED
                    )
                )

                # If a better match than the previous best, log that champion
                if champion_classify_percent > threshold:
                    threshold = champion_classify_percent
                    most_likely_champion = champion_name

            champions['red'][role]['champion'] = most_likely_champion
            if most_likely_champion != "":
                identified += 1

            print(f"Red {role} identified ({100 * threshold:.2f}%):  {most_likely_champion}")

        if identified < 10:
            for _ in range(frames_to_skip):
                cap.grab()

            ret, frame = cap.read()

    # Grab portraits of each champion found, to search for on the minimap
    for side in champions:
        for role in champions[side]:
            champion = champions[side][role]['champion']

            champion_minimap_icon = get_champion_minimap_icon(champion)

            champions[side][role]['template'] = champion_minimap_icon

    return champions

