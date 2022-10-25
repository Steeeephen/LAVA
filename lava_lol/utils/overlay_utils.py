from pathlib import Path
from typing import Generator, Tuple, Dict

import cv2
import numpy as np


def get_assets(folder) -> Generator[str, None, None]:
    """
    """
    assets_folder = (
        Path(__file__)
        .resolve()
        .parents[1] 
        / 'assets'
    )

    overlay_assets = (assets_folder / folder).glob('[!.]*')

    return overlay_assets


def get_summoner_spells() -> Dict[str, np.array]:
    """
    """
    summoner_folder = get_assets('summoners')
    summoners = {}

    for summoner_image_path in summoner_folder:
        summoner_image_path = str(summoner_image_path)

        summoners[summoner_image_path[:-4]] = cv2.imread(summoner_image_path, 0)

    return summoners


def get_champion_portraits(colour):
    """
    """
    if colour.lower() not in ['red', 'blue']:
        raise Exception(f'There is no {colour} team - only red and blue')

    champion_portrait_paths = get_assets(f'champ_classify_{colour.lower()}')

    champion_portraits = {}

    for champion_portrait_path in champion_portrait_paths:
        champion_name = champion_portrait_path.stem
        champion_portrait_path = str(champion_portrait_path)
 
        champion_portraits[champion_name] = cv2.imread(champion_portrait_path, 0)

    return champion_portraits


def get_champion_minimap_icon(champ) -> np.array:
    """
    """
    champion_minimap_icon_path = (
        Path(__file__)
        .resolve()
        .parents[1] 
        / 'assets'
        / 'champion_tracking'
        / f'{champ}.jpg'
    )

    if not champion_minimap_icon_path.exists():
        raise Exception(f'{champion_minimap_icon_path} not found')

    champion_minimap_icon = cv2.imread(
        str(champion_minimap_icon_path),
        0
    )

    return champion_minimap_icon


def get_header_borders(frame_height: int, 
                       frame_width: int) -> Tuple[slice, slice]:
    """
    """
    header_height = frame_height // 15
    header_width_left = 6 * (frame_width // 13)
    header_width_right = 7 * (frame_width // 13)

    header_borders = (
        slice(0, header_height),
        slice(header_width_left, header_width_right)
    )

    return header_borders


def get_first_ingame_frame(cap: cv2.VideoCapture,
                           header: np.array,
                           header_threshold: float = 0.8):
    """
    """
    ret, frame = cap.read()
    header_borders = get_header_borders(*frame.shape[:2])

    while ret is True:
        # Convert to grayscale
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        # Search only in a section near the top, again for efficiency
        cropped = gray[header_borders]

        # Look for the scoreboard
        matched = cv2.matchTemplate(
            cropped,
            header,
            cv2.TM_CCOEFF_NORMED
        )

        location = np.where(matched > header_threshold)

        if location[0].any():
            break

        # Skip one second if not
        for _ in range(frames_to_skip):
            cap.grab()

        ret, frame = cap.read()