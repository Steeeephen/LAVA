import cv2
import numpy as np
import pandas as pd

from ..utils.overlay_utils import get_header_borders


def track_champions(cap,
                    frames_to_skip,
                    map_coordinates,
                    header,
                    champs,
                    threshold: float = 0.65):
        H = map_coordinates[0].stop - map_coordinates[0].start
        W = map_coordinates[1].stop - map_coordinates[1].start

        data_entries = []

        ret, frame = cap.read()
        header_borders = get_header_borders(*frame.shape[:2])

        # digit_templates = utils.get_digit_templates(self.overlay)
        
        while ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cropped = gray[header_borders]

            matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
            location = np.where(matched > 0.65)

            if location[0].any():
                # second = read_timer(gray)

                # Crop to minimap
                cropped = gray[map_coordinates]

                for side in champs.keys():
                    for role in champs[side].keys():
                        template = champs[side][role]['template']

                        matched = cv2.matchTemplate(
                            cropped,
                            template,
                            cv2.TM_CCOEFF_NORMED
                        )
                        location = (np.where(matched == max(0.8, np.max(matched))))

                        # If champion found, save their location
                        try:
                            point = next(zip(*location[::-1]))
                            cv2.rectangle(
                                cropped,
                                point,
                                (point[0] + 14, point[1] + 14),
                                255,
                                2
                            )
                        except Exception:
                            point = (np.nan, np.nan)
                            pass

                        data_entries.append({
                            'champ': champs[side][role]['champion'],
                            'role': role,
                            'side': side,
                            'coords': np.array([(point[0] + 7) / H, (point[1] + 7) / W]),
                            # 'second': second
                        })

                # if show_minimap is True:
                    # Show minimap with champions highlighted
                cv2.imshow('minimap', cropped)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            for _ in range(frames_to_skip):
                cap.grab()
            ret, frame = cap.read()

        return pd.DataFrame(data_entries)
