import os
from typing import List

import cv2

def get_roles() -> List[str]:
    """
    """
    roles = [
        'top',
        'jgl',
        'mid',
        'adc',
        'sup'
    ]

    return roles

class Borders(object):
    """
    """
    ROLES = [
        'top',
        'jgl',
        'mid',
        'adc',
        'sup'
    ]

    # Dimensions of role portraits
    BLUE_PORTRAIT_BORDERS = {
        "top": (slice(108, 133), slice(20, 50)),
        "jgl": (slice(178, 203), slice(20, 50)),
        "mid": (slice(246, 268), slice(20, 50)),
        "adc": (slice(310, 340), slice(20, 50)),
        "sup": (slice(380, 410), slice(20, 50))
    }

    RED_PORTRAIT_BORDERS = {
        "top": (slice(108, 133), slice(1228, 1262)),
        "jgl": (slice(178, 203), slice(1228, 1262)),
        "mid": (slice(246, 268), slice(1228, 1262)),
        "adc": (slice(310, 340), slice(1228, 1262)),
        "sup": (slice(380, 410), slice(1228, 1262))
    }

    SUMMONER_SPELL_BORDERS = {
        "top": [slice(103, 121), slice(118, 137)],
        "jgl": [slice(172, 188), slice(187, 203)],
        "mid": [slice(241, 257), slice(256, 272)],
        "adc": [slice(309, 325), slice(324, 340)],
        "sup": [slice(378, 394), slice(393, 409)],
        "blue": slice(1, 19),
        "red": slice(1261, 1279),
    }

    TIMER_BORDERS = {
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

    MINIMAP_BORDERS = {
        "lec_summer_2020": [563, 712, 1108, 1258],
        "lec_spring_2021": [563, 712, 1103, 1259],
        "lec_summer_2021": [569, 706, 1107, 1257]
    }