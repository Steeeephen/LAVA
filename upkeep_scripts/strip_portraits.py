import argparse
from pathlib import Path
from textwrap import dedent

import cv2
import youtubesearchpython as yt


def main(url, local):
    """
    Helper script to pull champion portraits from the sidebar

    Arguments:
    ----------
    url : string
        Link to Youtube video or path to local video
    local : bool
        Use local videos instead of Youtube
    """
    print("Welcome to the helper script for saving champion portraits from a video")

    # Load URL
    if local is False:
        fetcher = yt.StreamURLFetcher()

        video = yt.Video.get(url)
        url = fetcher.get(video, 22)

    cap = cv2.VideoCapture(url)

    print(
        "Video Loaded. "
        "Iterate through frames until you find all champions at level one"
    )

    # Find the right frame
    while True:
        frame_input = input("Show Frame Number (type 'q' to quit): ")

        if frame_input.lower() in ('q', 'quit'):
            break

        # Sets video to frame
        cap.set(1, int(frame_input))
        ret, frame = cap.read()

        cv2.imshow("Is the champion isolated?", frame)
        cv2.waitKey()

    cv2.destroyAllWindows()

    # Default locations of the sidebar portraits
    locations = {
        'vertical': [110, 180, 247, 315, 383],
        'blue': slice(25, 45),
        'red': slice(1235, 1255)
    }

    print(dedent("""
        Input Numbers:
        1  : Blue side toplaner
        2  : Blue side jungler
        3  : Blue side midlaner
        4  : Blue side ADC
        5  : Blue side support

        6  : Red side toplaner
        7  : Red side jungler
        8  : Red side midlaner
        9  : Red side ADC
        10 : Red side support
    """))

    # Keep adding portraits
    while True:
        frame_input = input("Number (type 'q' to exit):\n")

        if frame_input.lower() in ('q', 'quit'):
            break

        # Champion name
        name = input("Champ Name:\n")

        idx = int(frame_input) - 1
        vertical = slice(
            locations['vertical'][idx % 5],
            locations['vertical'][idx % 5] + 20,
        )

        col = "blue" if idx < 5 else 'red'
        horizontal = locations[col]

        # Crop image to portrait
        cropped = frame[vertical, horizontal]

        path = (
            Path(__file__).resolve().parent.parent
            / 'assets'
            / 'tracking'
            / 'champ_classifying'
            / col
            / f'{name}.jpg'
        )

        gray_cropped = cv2.cvtColor(
            cropped,
            cv2.COLOR_BGR2GRAY
        )

        # Save image to directory
        cv2.imwrite(str(path), gray_cropped)
        print(f"{name} saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helper script for saving templates from a video")

    parser.add_argument('-l', '--local', action='store_true',
                        help='Use local videos instead of Youtube')
    parser.add_argument('-u', '--url', type=str, required=True,
                        help='Link to Youtube video or path to local video')

    args = parser.parse_args()

    main(**vars(args))
