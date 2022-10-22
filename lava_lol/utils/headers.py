from typing import Tuple

import cv2
import numpy as np

from utils.overlay_utils import get_assets, get_header_borders

def get_headers(cap: cv2.VideoCapture,
                frames_to_skip: int, 
                threshold: float = 0.7) -> Tuple[int, np.array, str]:
    """
    """
    headers_folder = get_assets('headers')

    header_templates = {}
    for header in headers_folder:
        header = str(header)
        header_templates[header] = cv2.imread(header, 0)

    ret, frame = cap.read()
    header_borders = get_header_borders(*frame.shape[:2])

    header_found = False
    count = 0

    while ret is True and header_found is False:
        count += 1

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )
        
        cropped = gray[header_borders]

        for header in header_templates.keys():
            header_max_probability = np.max(
                cv2.matchTemplate(
                    cropped,
                    header_templates[header],
                    cv2.TM_CCOEFF_NORMED)
            )

            if header_max_probability > threshold:
                header_found = True
                header_temp = header
                threshold = header_max_probability

        cap.set(1, frames_to_skip * 120 * count)
        ret, frame = cap.read()

        if header_found is True:
            header = header_templates[header_temp]
            overlay = header_temp.split('.')[0]
            return count, header, overlay

    raise Exception('Header not found in video overlay')