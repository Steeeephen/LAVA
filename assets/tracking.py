import cv2
import pandas as pd
import numpy as np

from assets.constants import *

def tracker(champs, header, cap, templates,
            map_coordinates, frames_to_skip, minimap):  # noq-E501
    _, frame = cap.read()
    hheight, hwidth, _ = frame.shape
    hheight = hheight//15
    hwidth1 = 6*(hwidth//13)
    hwidth2 = 7*(hwidth//13)

    H = map_coordinates[1] - map_coordinates[0]
    W = map_coordinates[3] - map_coordinates[2]

    data_entries = []

    roles = list(role_dict.keys())

    count = 1
    ret, frame = cap.read()
        
    while ret is True:
        count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cropped = gray[0:hheight, hwidth1:hwidth2]

        matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
        location = np.where(matched > 0.65)
        if location[0].any():
            cropped = gray[35:70, 625:665]
            digits = []

            for num in digit_templates.keys():
                template = digit_templates[num]
                check = cv2.matchTemplate(
                    cropped, template, cv2.TM_CCOEFF_NORMED)
                digit = np.where(check > 0.85)
                for test in (list(zip(*digit[::-1]))):
                    digits.append([test, num])

            digits.sort()
            try:
                second = 600*digits[0][1]+60 * \
                    digits[1][1]+10*digits[2][1]+digits[3][1]
            except IndexError:
                second = np.nan
            
            # Crop to minimap and Baron Nashor icon
            cropped = gray[map_coordinates[0]: map_coordinates[1],
                           map_coordinates[2]: map_coordinates[3]]

            for template_i, template in enumerate(templates):

                matched = cv2.matchTemplate(
                    cropped, template, cv2.TM_CCOEFF_NORMED)
                location = (np.where(matched == max(0.8, np.max(matched))))

                # If champion found, save their location
                #############
                try:
                    point = next(zip(*location[::-1]))
                    cv2.rectangle(cropped, point,
                                  (point[0] + 14, point[1] + 14), 255, 2)
                except:
                    point = [np.nan, np.nan]
                    pass

                side = 'blue' if template_i <= 4 else 'red'
                role = roles[template_i % 5]

                data_entries.append({
                    'champ': champs[template_i],
                    'role': role, 
                    'side': side, 
                    'x': (point[0] + 7)/H,
                    'y': (point[1] + 7)/W, 
                    'second': second
                })

            if minimap is True:
                # Show minimap with champions highlighted
                cv2.imshow('minimap', cropped)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        for _ in range(frames_to_skip):
            cap.grab()
        ret, frame = cap.read()

    return pd.DataFrame(data_entries)
