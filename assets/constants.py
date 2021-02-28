import cv2
import os

tracking_folder = os.path.join(
    "assets",
    "tracking")

digits_path = os.path.join(
    tracking_folder,
    "digits")

baron_image = os.path.join(
    tracking_folder,
    "baron.jpg")

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

# Dimensions of in-game timer
baron = [23, 50, 1207, 1230]

# Baron spawns at 20mins, when it appears we use this to sync the time
baron_template = cv2.imread(baron_image, 0)

# Dimensions of role portraits
role_dict = {
    "top": [108, 133],
    "jgl": [178, 203],
    "mid": [246, 268],
    "adc": [310, 340],
    "sup": [380, 410]}

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
