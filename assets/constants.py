import cv2
import os

tracking_folder = os.path.join(
    "assets",
    "tracking")

digits_path = os.path.join(
    tracking_folder,
    "digits")

blue_classify_path = os.path.join(
    tracking_folder,
    "champ_classifying",
    "blue"
)

red_classify_path = os.path.join(
    tracking_folder,
    "champ_classifying",
    "red"
)

# Dimensions of role portraits
blue_champ_sidebar = {
    "top": (slice(108, 133), slice(20, 50)),
    "jgl": (slice(178, 203), slice(20, 50)),
    "mid": (slice(246, 268), slice(20, 50)),
    "adc": (slice(310, 340), slice(20, 50)),
    "sup": (slice(380, 410), slice(20, 50))}

red_champ_sidebar = {
    "top": (slice(108, 133), slice(1228, 1262)),
    "jgl": (slice(178, 203), slice(1228, 1262)),
    "mid": (slice(246, 268), slice(1228, 1262)),
    "adc": (slice(310, 340), slice(1228, 1262)),
    "sup": (slice(380, 410), slice(1228, 1262))}

timer_borders = (
    slice(35,70),
    slice(625,665))

leagues = {
    "lec_summer_2020": [563, 712, 1108, 1258]
}

# The digits for reading the timer
digit_templates = dict()

# won't be run if constants imported from another directory
if os.path.exists(digits_path):
    for digit in range(10):
        digit_image_path = os.path.join(
            digits_path,
            f'{digit}.jpg'
        )

        digit_templates[digit] = cv2.imread(digit_image_path, 0)

    blue_portraits = os.listdir(blue_classify_path)
    red_portraits = os.listdir(red_classify_path)

    blue_champ_templates = [""]*len(blue_portraits)
    red_champ_templates = [""]*len(red_portraits)

    # Save templates for template matching
    for portrait_i, portrait in enumerate(blue_portraits):
        portrait_image = os.path.join(
            blue_classify_path,
            portrait)

        blue_champ_templates[portrait_i] = cv2.imread(portrait_image, 0)

    for portrait_i, portrait in enumerate(red_portraits):
        portrait_image = os.path.join(
            red_classify_path,
            portrait)

        red_champ_templates[portrait_i] = cv2.imread(portrait_image, 0)
