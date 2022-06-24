import argparse
from pathlib import Path

import cv2
import youtubesearchpython as yt


def main(url: str, local: bool) -> None:
    """
    Helper script to make it easier to add templates for tracking

    Arguments:
    ----------
    url : string
        Link to Youtube video or path to local video
    local : bool
        Use local videos instead of Youtube
    """
    print("Welcome to the helper script for saving templates from a video")

    # Load URL
    if local is False:
        fetcher = yt.StreamURLFetcher()

        video = yt.Video.get(url)
        url = fetcher.get(video, 22)

    cap = cv2.VideoCapture(url)

    print(
        "Video Loaded. "
        "Iterate through frames until you find the champion isolated"
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

    champ = input("Champion Name: ")

    path = (
        Path(__file__).resolve().parent.parent
        / 'assets'
        / 'tracking'
        / 'champ_tracking'
        / f'{champ}.jpg'
    )

    print(
        "Now you want to get a 14x14 square of the champion,",
        "iterate through bounding box values until you get it.",
        "\nNote: Generally the minimap is within (550, 720, 1100, 1260)"
    )

    # Isolate champion
    while True:
        input_0 = input("Y-coordinate upper: (type 'q' to quit) ")

        if str(input_0).lower() in ('q', 'quit'):
            break

        input_0 = int(input_0)
        input_1 = int(input("Y-coordinate lower: "))
        input_2 = int(input("X-coordinate left: "))
        input_3 = int(input("X-coordinate right: "))

        # Crop frame to inputs
        cropped = frame[
            input_0:input_1,
            input_2:input_3
        ]

        cv2.destroyAllWindows()

        cv2.imshow("Crop", cropped)
        cv2.waitKey()

    # Convert to grayscale
    gray_cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    # Saves image in correct directory
    cv2.imwrite(
        str(path),
        gray_cropped
    )

    print(f'{champ} saved to {path}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Helper script for saving templates from a video")

    parser.add_argument('-l', '--local', action='store_true',
                        help='Use local videos instead of Youtube')
    parser.add_argument('-u', '--url', type=str, required=True,
                        help='Link to Youtube video or path to local video')

    args = parser.parse_args()

    main(**vars(args))
