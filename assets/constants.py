import os

import cv2

tracking_folder = os.path.join(
    "assets",
    "tracking")

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

roles = ['top', 'jgl', 'mid', 'adc', 'sup']

summoner_path = os.path.join(
    tracking_folder,
    'summoners'
)

summoners = {}

for summoner in os.listdir(summoner_path):
    summoner_image_path = os.path.join(
        summoner_path,
        summoner
    )
    summoners[summoner[:-4]] = cv2.imread(summoner_image_path, 0)

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

summoner_spells = {
    "top": [slice(103, 121), slice(118, 137)],
    "jgl": [slice(172, 188), slice(187, 203)],
    "mid": [slice(241, 257), slice(256, 272)],
    "adc": [slice(309, 325), slice(324, 340)],
    "sup": [slice(378, 394), slice(393, 409)],
    "blue": slice(1, 19),
    "red": slice(1261, 1279),
}

timer_borders = {
    'default': {
        'minute': (
            slice(52, 63),
            slice(628, 641)
        ),
        'second': (
            slice(52, 63),
            slice(644, 658)
        )
    }
}

leagues = {
    "lec_summer_2020": [563, 712, 1108, 1258],
    "lec_spring_2021": [563, 712, 1103, 1259],
    "lec_summer_2021": [569, 706, 1107, 1257]
}

# won't be run if constants imported from another directory
if os.path.exists(blue_classify_path):
    blue_portraits = os.listdir(blue_classify_path)
    red_portraits = os.listdir(red_classify_path)

    blue_champ_templates = [""] * len(blue_portraits)
    red_champ_templates = [""] * len(red_portraits)

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
