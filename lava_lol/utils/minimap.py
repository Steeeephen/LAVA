import cv2
import numpy as np

from .overlay_utils import get_assets
from .constants import Borders


def get_map_borders(cap: cv2.VideoCapture,
                    league: str,
                    frames_to_skip: int):
    """
    """
    inhib_templates = []

    if league != 'auto':
        borders = Borders()
        borders = borders.MINIMAP_BORDERS[league]

        return (
            slice(borders[0], borders[1]),
            slice(borders[2], borders[3])
        )
    
    inhib_template_paths = get_assets('minimap')

    for inhib_template_path in inhib_template_paths:
        inhib_templates.append(cv2.imread(str(inhib_template_path), 0))

    inhibs_found = 0

    ret, frame = cap.read()

    height, width, _ = frame.shape

    crop_frame_corner = (
        slice(2 * height // 3, height),
        slice(4 * width // 5, width)
    )

    while ret is True and inhibs_found < 4:
        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        cropped = gray[crop_frame_corner]

        cv2.imshow('ok', cropped)
        cv2.waitKey()

        inhib_locations = []
        inhibs_found = 0

        for inhib in inhib_templates:
            res = cv2.matchTemplate(
                cropped,
                inhib,
                cv2.TM_CCOEFF_NORMED
            )

            location = (np.where(res == max(0.65, np.max(res))))

            try:
                point = next(zip(*location[::-1]))
                inhibs_found += 1
                print('inhib found')
            except Exception:
                inhibs_found = 0
                print("Not all inhibs found")

                for _ in range(frames_to_skip):
                    cap.grab()
                    ret, frame = cap.read()
                break
            inhib_locations.append(point)

        if inhibs_found == 4:
            swapped_inhibs = [
                coord[::-1] for coord in
                inhib_locations[2:] + inhib_locations[:2]]

            inhib_order = inhib_locations != sorted(inhib_locations)
            swapped_order = swapped_inhibs != sorted(swapped_inhibs)
            if inhib_order is True or swapped_order is True:
                print('Inhibs found in incorrect order')

                inhibs_found = 0
                for _ in range(frames_to_skip):
                    cap.grab()
                ret, frame = cap.read()

    inhib_blue_top = inhib_locations[0]
    inhib_blue_bot = inhib_locations[1]
    inhib_red_top = inhib_locations[2]
    inhib_red_bot = inhib_locations[3]

    dist2 = np.linalg.norm(
        np.array(inhib_blue_top) - np.array(inhib_blue_bot))
    dist4 = np.linalg.norm(
        np.array(inhib_red_bot) - np.array(inhib_red_top))

    inhib_blue_top = (
        (inhib_blue_bot[0] - inhib_blue_top[0]) // 4 + inhib_blue_top[0],
        (inhib_blue_bot[1] - inhib_blue_top[1]) // 4 + inhib_blue_top[1]
    )

    x = Point(inhib_blue_top[0], inhib_blue_top[1])
    y = Point(inhib_blue_bot[0], inhib_blue_bot[1])

    c1 = Circle(x, dist2)
    c2 = Circle(y, dist2)
    points = np.array(
        [np.array([N(i).x, N(i).y]) for i in intersection(c1, c2)]
    )

    intersect1 = points[np.argmin(points[:, 1])]

    inhib_red_top = (
        (inhib_red_bot[0] - inhib_red_top[0]) // 4 + inhib_red_top[0],
        (inhib_red_bot[1] - inhib_red_top[1]) // 4 + inhib_red_top[1]
    )

    x = Point(inhib_red_bot[0], inhib_red_bot[1])
    y = Point(inhib_red_top[0], inhib_red_top[1])

    c3 = Circle(x, dist4)
    c4 = Circle(y, dist4)

    points = np.array(
        [np.array([N(i).x, N(i).y]) for i in intersection(c3, c4)])

    intersect2 = points[np.argmin(points[:, 0])]

    line = Line(intersect1, intersect2)

    points = np.array(
        [np.array([N(i).x, N(i).y]) for i in intersection(line, c1)])
    border1 = points[np.argmin(points[:, 0])]

    points = np.array(
        [np.array([N(i).x, N(i).y]) for i in intersection(line, c4)])
    border2 = points[np.argmin(points[:, 1])]

    return (
        slice(
            height // 2 + int(border2[1]),
            height // 2 + int(border1[1])),
        slice(
            4 * width // 5 + int(border1[0]),
            4 * width // 5 + int(border2[0]))
    )