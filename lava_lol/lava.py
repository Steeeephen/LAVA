import os

import cv2
import numpy as np
import pandas as pd
from numpy.linalg import norm
from sympy.geometry import Point, Circle, intersection, Line
from sympy import N
from assets.utils import utils
from assets.utils.ytHelper import is_valid_youtube_url, parse_youtube_url
from assets.utils.video_file_helper import parse_local_files

from assets.graphing import GraphsOperator
import assets.utils.constants as constants

class LAVA(GraphsOperator):
    def __init__(self):
        text_recogniser_path = os.path.join(
            'assets',
            'tracking',
            'crnn.onnx'
        )

        self.recogniser = cv2.dnn.readNet(text_recogniser_path)
        self.recogniser.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.recogniser.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def execute(self,
                url="",
                output_file='positions.csv',
                minimap=False,
                graphs=False,
                lightweight=True):
        df = self.gather_data(
            url,
            output_file,
            minimap,
            lightweight
        )

        if graphs is True:
            self.draw_graphs(df, df.video.iloc[0])
        
        print(f'Video {df.video.iloc[0]} complete')

    def gather_data(self,
                    url="",
                    output_file='positions.csv',
                    minimap=False,
                    lightweight=True):
        videos = self.parse_url(url)
        full_data = pd.DataFrame()

        self.lightweight = lightweight

        for video, url in videos.items():
            print(f'Running Video: {video}')
            self.cap = cv2.VideoCapture(url)
            self.frames_to_skip = int(self.cap.get(cv2.CAP_PROP_FPS))

            try:
                self.cap.set(
                    1,
                    self.frames_to_skip * int(self.url.split('t=')[1])
                )
            except Exception:
                # tbfixed
                pass

            self.minimap = minimap

            # headers
            count = self.headers()

            if count is None:
                continue

            self.cap.set(1, self.frames_to_skip * 120 * (count - 2))

            champs = self.identify(self.cap)
            print("Champs identified")

            self.match_summoners = self.summoner_spells(self.cap)
            print('Summoner Spells identified')

            # map borders
            map_coordinates = self.map_borders(self.cap, league='automatic')
            print(f"Map coordinates found: {','.join(map(str, map_coordinates))}")

            print("Tracking commenced")
            df = self.tracker(map_coordinates, champs)
            print("Tracking complete")

            df = self.interpolate(df)
            print("Interpolation completed")

            df['video'] = video

            # Champs with updated splash arts are denoted with e.g. 'udyr_' and 'udyr'
            df['champ'] = df['champ'].str.replace('_', '')

            df.to_csv(
                output_file,
                index=False
            )

            full_data = full_data.append(df)
            cv2.destroyAllWindows()

        return full_data
    def gather_info(self, url=""):
        videos = self.parse_url(url)

        full_data = {}

        for video, url in videos.items():
            self.cap = cv2.VideoCapture(url)
            self.frames_to_skip = int(self.cap.get(cv2.CAP_PROP_FPS))

            try:
                self.cap.set(1, self.frames_to_skip * int(self.url.split('t=')[1]))
            except Exception:
                pass

            count = self.headers()

            if count is None:
                continue

            self.cap.set(1, self.frames_to_skip * 120 * (count - 2))

            champs = self.identify(self.cap)

            for side in champs.keys():
                for role in champs[side].keys():
                    champs[side][role] = champs[side][role]['champ']

            full_data[video] = champs
        return full_data

    def parse_url(self, url=""):
        if is_valid_youtube_url(url):
            return parse_youtube_url(url)
        else:
            return parse_local_files(url)

    def headers(self):
        headers_path = os.path.join(
            constants.tracking_folder,
            'headers'
        )

        headers_list = os.listdir(headers_path)

        header_templates = {}
        for header in headers_list:
            header_directory = os.path.join(
                headers_path,
                header
            )

            header_templates[header] = cv2.imread(header_directory, 0)

        ret, frame = self.cap.read()
        header_borders = utils.get_header_borders(frame.shape)

        header_found = False
        threshold = 0.8
        count = 0

        while ret is True and header_found is False:
            count += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

            self.cap.set(1, self.frames_to_skip * 120 * count)
            ret, frame = self.cap.read()

            if header_found is True:
                print(f'{header_temp} header collected after {count*120} seconds')
                self.header = header_templates[header_temp]
                self.overlay = header_temp.split('.')[0]
                return count

    def identify(self, cap):
        ret, frame = cap.read()
        header_borders = utils.get_header_borders(frame.shape)

        self.roles = constants.blue_champ_sidebar.keys()

        # Instantiate dictionary for champions
        champs = {'blue': {}, 'red': {}}

        for role in self.roles:
            champs['blue'][role] = {'champ': "", 'template': ""}
            champs['red'][role] = {'champ': "", 'template': ""}

        # Grab portraits for identifying the champions played
        identified = 0

        # Identify blue side champions until exactly 5 have been found
        while identified != 5:
            while ret is True:
                # Making the images gray will make template matching more efficient
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Search only in a section near the top, again for efficiency
                cropped = gray[header_borders]

                # Look for the scoreboard
                matched = cv2.matchTemplate(cropped, self.header, cv2.TM_CCOEFF_NORMED)
                location = np.where(matched > 0.75)
                if location[0].any():
                    break

                # Skip one second if not
                for _ in range(self.frames_to_skip):
                    cap.grab()

                ret, frame = cap.read()
            identified = 0

            self.roles = constants.blue_champ_sidebar.keys()

            # Check the sidebar for each champion
            for role in self.roles:
                temp = 0.65
                most_likely_champ = ""

                sidebar_borders = constants.blue_champ_sidebar[role]

                # Crop to each role's portrait in the video
                blue_crop = gray[sidebar_borders]

                # Scroll through each template and find the best match
                for j, template in enumerate(constants.blue_champ_templates):
                    champ_classify_percent = np.max(
                        cv2.matchTemplate(
                            blue_crop,
                            template,
                            cv2.TM_CCOEFF_NORMED
                        )
                    )

                    # If a better match than the previous best, log that champion
                    if champ_classify_percent > temp:
                        temp = champ_classify_percent
                        most_likely_champ = constants.blue_portraits[j][:-4]

                print(
                    f"Blue {role} identified ({100*temp:.2f}%):  {most_likely_champ.capitalize()}"
                )
                champs['blue'][role]['champ'] = most_likely_champ
                if most_likely_champ != "":
                    identified += 1

            # If too few champions found, skip a frame and try again
            if identified < 5:
                for _ in range(self.frames_to_skip):
                    cap.grab()

                _, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print("Too few champions found")

        # Same for red side champions
        while identified != 10:
            while ret is True:
                # Making the images gray will make template matching more efficient
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Search only in a section near the top, again for efficiency
                cropped = gray[header_borders]

                # Look for the scoreboard
                matched = cv2.matchTemplate(cropped, self.header, cv2.TM_CCOEFF_NORMED)
                location = np.where(matched > 0.75)
                if location[0].any():
                    break

                # Skip one second if not
                for _ in range(self.frames_to_skip):
                    cap.grab()

                ret, frame = cap.read()
            identified = 5

            for role in self.roles:
                temp = 0.65
                most_likely_champ = ""

                sidebar_borders = constants.red_champ_sidebar[role]

                red_crop = gray[sidebar_borders]

                for j, template in enumerate(constants.red_champ_templates):
                    champ_classify_percent = np.max(cv2.matchTemplate(
                        red_crop, template, cv2.TM_CCOEFF_NORMED))

                    if champ_classify_percent > temp:
                        temp = champ_classify_percent
                        most_likely_champ = constants.  red_portraits[j][:-4]

                print(
                    f"Red {role} identified ({100*temp:.2f}%):  {most_likely_champ.capitalize()}"
                )
                champs['red'][role]['champ'] = most_likely_champ
                if most_likely_champ != "":
                    identified += 1

            if identified < 10:
                for _ in range(self.frames_to_skip):
                    cap.grab()

                ret, frame = cap.read()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print("Too few champions found")

        # Grab portraits of each champion found, to search for on the minimap
        for side in champs:
            for role in champs[side]:
                champ = champs[side][role]['champ']

                champ_image = os.path.join(
                    constants.tracking_folder,
                    'champ_tracking',
                    f'{champ}.jpg')
                champs[side][role]['template'] = cv2.imread(champ_image, 0)

        return champs

    def summoner_spells(self, cap):
        _, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        match_summoners = {
            'red': {k: [] for k in self.roles},
            'blue': {k: [] for k in self.roles}
        }

        for side in ['red', 'blue']:
            for role in self.roles:
                for i in range(2):
                    cropped = gray[
                        constants.summoner_spells[role][i],
                        constants.summoner_spells[side]]
                    threshold = 0.

                    for summoner in constants.summoners.keys():
                        max_probability = np.max(
                            cv2.matchTemplate(
                                cropped,
                                constants.summoners[summoner],
                                cv2.TM_CCOEFF_NORMED)
                        )

                        if max_probability > threshold:
                            summoner_temp = summoner
                            threshold = max_probability

                    match_summoners[side][role].append(summoner_temp)

        return match_summoners

    def map_borders(self, cap, league):
        inhib_templates = []

        if league == 'automatic':
            for i in range(4):
                inhib_path = os.path.join(
                    constants.tracking_folder,
                    "minimap",
                    f"inhib{i}.jpg"
                )

                inhib_templates.append(cv2.imread(inhib_path, 0))

            inhibs_found = 0
            ret, frame = cap.read()
            height, width, _ = frame.shape

            while ret is True and inhibs_found < 4:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                crop_frame_corner = (
                    slice(height // 2, height),
                    slice(4 * width // 5, width)
                )

                cropped = gray[crop_frame_corner]

                inhib_locations = []
                inhibs_found = 0

                for inhib in inhib_templates:
                    res = cv2.matchTemplate(cropped, inhib, cv2.TM_CCOEFF_NORMED)
                    location = (np.where(res == max(0.65, np.max(res))))

                    try:
                        point = next(zip(*location[::-1]))
                        inhibs_found += 1
                    except Exception:
                        inhibs_found = 0
                        print("Not all inhibs found")

                        for i in range(self.frames_to_skip):
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
                        for i in range(self.frames_to_skip):
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
        else:
            borders = constants.leagues[league]

            return (
                slice(borders[0], borders[1]),
                slice(borders[2], borders[3])
            )

    def read_timer(self, gray):
        if self.lightweight is True:
            # to be changed, this is the default
            cropped_timer = gray[23:50, 1210:1250]
            nums = {}
            for image_i in self.digit_templates:
                res = cv2.matchTemplate(
                    cropped_timer,
                    self.digit_templates[image_i], cv2.TM_CCOEFF_NORMED
                )

                # Try to find each digit in the timer
                digit = np.where(res > 0.75)
                if(digit[0].any()):
                    seen = set()
                    inp = (list(zip(*digit)))
                    outp = [
                        (a, b) for a, b in inp
                        if not (
                            b in seen
                            or seen.add(b)
                            or seen.add(b + 1)
                            or seen.add(b - 1)
                        )
                    ]
                    for out in outp:
                        nums[out[1]] = image_i
            timer_ordered = ""

            # Sort time
            for num in (sorted(nums)):
                timer_ordered = ''.join([timer_ordered, nums[num]])
            try:
                seconds = int(utils.timer(timer_ordered))
            except ValueError:
                seconds = np.nan
        else:
            cropped = gray[self.minute_borders]

            blob = cv2.dnn.blobFromImage(
                cropped,
                size=(100, 32),
                mean=127.5,
                scalefactor=1 / 127.5
            )
            self.recogniser.setInput(blob)

            minute = self.recogniser.forward()

            cropped = gray[self.second_borders]

            blob = cv2.dnn.blobFromImage(
                cropped,
                size=(100, 32),
                mean=127.5,
                scalefactor=1 / 127.5
            )
            self.recogniser.setInput(blob)

            second = self.recogniser.forward()

            try:
                seconds = 60 * int(utils.decodeText(minute)) + int(utils.decodeText(second))
            except ValueError:
                seconds = np.nan

        return seconds

    def tracker(self,
                map_coordinates,
                champs):
        H = map_coordinates[0].stop - map_coordinates[0].start
        W = map_coordinates[1].stop - map_coordinates[1].start

        data_entries = []

        ret, frame = self.cap.read()
        header_borders = utils.get_header_borders(frame.shape)

        if self.lightweight is True:
            self.digit_templates = utils.get_digit_templates(self.overlay)
        else:
            self.minute_borders = constants.timer_borders[self.overlay]['minute']
            self.second_borders = constants.timer_borders[self.overlay]['second']
        while ret is True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cropped = gray[header_borders]

            matched = cv2.matchTemplate(cropped, self.header, cv2.TM_CCOEFF_NORMED)
            location = np.where(matched > 0.65)

            if location[0].any():
                second = self.read_timer(gray)

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
                            'champ': champs[side][role]['champ'],
                            'role': role,
                            'side': side,
                            'coords': np.array([(point[0] + 7) / H, (point[1] + 7) / W]),
                            'second': second,
                            'summoner_spell_1': self.match_summoners[side][role][0],
                            'summoner_spell_2': self.match_summoners[side][role][1]
                        })

                if self.minimap is True:
                    # Show minimap with champions highlighted
                    cv2.imshow('minimap', cropped)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            for _ in range(self.frames_to_skip):
                self.cap.grab()
            ret, frame = self.cap.read()

        return pd.DataFrame(data_entries)
