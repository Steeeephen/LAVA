import cv2
import numpy as np
import pandas as pd
import os
from sympy.geometry import Point, Circle, intersection, Line
from sympy import N

from assets.constants import *

def clean_for_directory(video_title):
    for ch in ['*', '.', '"', '/', '\\', ':', ';', '|', ',']:
        if ch in video_title:
            video_title = video_title.replace(ch, '')

    return video_title

def headers(cap, frames_to_skip, logger):
    headers_list = os.listdir('assets/tracking/headers')
    header_templates = {}

    for header in headers_list:
        header_directory = os.path.join(
            'assets/tracking/headers',
            header
        )
        header_templates[header] = cv2.imread(header_directory, 0)

    ret, frame = cap.read()
    frame_height, frame_width, _ = frame.shape
    header_height = frame_height//15
    header_width_left = 6*(frame_width//13)
    header_width_right = 7*(frame_width//13)

    header_found = False
    count = 0
    while(ret and not header_found):
        count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped = gray[0:header_height, header_width_left:header_width_right]

        for header in header_templates.keys():
            matched = cv2.matchTemplate(cropped, header_templates[header], cv2.TM_CCOEFF_NORMED)
            location = np.where(matched > 0.75)
            if(location[0].any()):
                logger.info(f'Header Found: {header}')
                header_found = True
                break

        cap.set(1, frames_to_skip*120*count)
        ret, frame = cap.read()
    return frame_height, frame_width, header_templates[header], count


def map_borders(cap, frames_to_skip, header, frame_height, frame_width):
    inhib_templates = []

    for i in range(4):
        inhib_templates.append(cv2.imread(
            'assets/tracking/minimap/inhib%d.jpg' % i, 0))

    inhibs_found = 0
    ret, frame = cap.read()
    h, w, _ = frame.shape

    while ret is True and inhibs_found < 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cropped = gray[h//2:h, (4*w)//5:w]

        inhib_locations = []
        inhibs_found = 0

        for inhib in inhib_templates:
            res = cv2.matchTemplate(cropped, inhib, cv2.TM_CCOEFF_NORMED)
            location = (np.where(res == max(0.65, np.max(res))))

            try:
                point = next(zip(*location[::-1]))
                inhibs_found += 1
                cv2.rectangle(
                    cropped, point, (point[0] + 18, point[1] + 12), 255, 2)
            except:
                inhibs_found = 0
                logger.info("Not all inhibs found")
                for i in range(frames_to_skip):
                    cap.grab()
                break
            inhib_locations.append(point)

        if inhibs_found == 4:
            swapped_inhibs = [
                coord[::-1] for coord in
                inhib_locations[2:] + inhib_locations[:2]]

            inhib_order = inhib_locations != sorted(inhib_locations)
            swapped_order = swapped_inhibs != sorted(swapped_inhibs)
            if inhib_order is True or swapped_order is True:
                logger.info('Inhibs not in correct order')
                
                inhibs_found = 0
                for i in range(frames_to_skip):
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
        (inhib_blue_bot[0] - inhib_blue_top[0])//4 + inhib_blue_top[0],
        (inhib_blue_bot[1] - inhib_blue_top[1])//4 + inhib_blue_top[1]
    )

    x = Point(inhib_blue_top[0], inhib_blue_top[1])
    y = Point(inhib_blue_bot[0], inhib_blue_bot[1])

    c1 = Circle(x, dist2)
    c2 = Circle(y, dist2)
    points = np.array(
      [np.array([N(i).x, N(i).y]) for i in intersection(c1, c2)])
    
    intersect1 = points[np.argmin(points[:, 1])]

    inhib_red_top = (
        (inhib_red_bot[0] - inhib_red_top[0])//4 + inhib_red_top[0],
        (inhib_red_bot[1] - inhib_red_top[1])//4 + inhib_red_top[1]
    )

    x = Point(inhib_red_bot[0], inhib_red_bot[1])
    y = Point(inhib_red_top[0], inhib_red_top[1])

    c3 = Circle(x, dist4)
    c4 = Circle(y, dist4)
    
    points = np.array(
      [np.array([N(i).x, N(i).y]) for i in intersection(c3, c4)])
    
    intersect2 = points[np.argmin(points[:, 0])]

    l = Line(intersect1, intersect2)
    
    points = np.array(
      [np.array([N(i).x, N(i).y]) for i in intersection(l, c1)])
    border1 = points[np.argmin(points[:, 0])]

    points = np.array(
      [np.array([N(i).x, N(i).y]) for i in intersection(l, c4)])
    border2 = points[np.argmin(points[:, 1])]

    cropped = cropped[
      int(border2[1]):int(border1[1]), int(border1[0]):int(border2[0])]
    
    return [
        frame_height//2+int(border2[1]),
        frame_height//2+int(border1[1]),
        (4*frame_width)//5 + int(border1[0]),
        (4*frame_width)//5 + int(border2[0])
    ]
    


def identify(cap, frames_to_skip, header, frame_height, frame_width, logger):
    ret, frame = cap.read()
    header_height = frame_height//15
    header_width_left = 6*(frame_width//13)
    header_width_right = 7*(frame_width//13)

    # Templates stores the champion pictures to search for
    templates = [0]*10

    while ret is True:
        ret, frame = cap.read()

        # Making the images gray will make template matching more efficient
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Search only in a section near the top, again for efficiency
        cropped = gray[0:header_height, header_width_left:header_width_right]

        # Look for the scoreboard
        matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
        location = np.where(matched > 0.75)
        if location[0].any():
            break

        # Skip one second if not
        for _ in range(frames_to_skip):
            cap.grab()

    # Grab portraits for identifying the champions played
    identified = 0

    # Identify blue side champions until exactly 5 have been found
    while identified != 5:
        identified = 0
        champs = [""]*10

        # Check the sidebar for each champion
        for role_i, role in enumerate(['top', 'jgl', 'mid', 'adc', 'sup']):
            temp = 0.7
            most_likely_champ = ""
            champ_found = False

            # Crop to each role's portrait in the video
            blue_crop = gray[role_dict[role]
                             [0]:role_dict[role][1], 20:50]

            # Scroll through each template and find the best match
            for j, template in enumerate(blue_champ_templates):
                champ_classify_percent = np.max(cv2.matchTemplate(
                    blue_crop, template, cv2.TM_CCOEFF_NORMED))

                # If a better match than the previous best, log that champion
                if(champ_classify_percent > temp):
                    champ_found = True
                    temp = champ_classify_percent
                    most_likely_champ = blue_portraits[j][:-4]
            logger.info(f"Blue {role} identified ({100*temp:.2f}%):  {most_likely_champ.capitalize()}")
            champs[role_i] = most_likely_champ
            identified += champ_found

        # If too few champions found, skip a frame and try again
        if identified < 5:
            for _ in range(frames_to_skip):
                cap.grab()
            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            logger.info("Too few champions found")

    # Same for red side champions
    while identified != 10:
        identified = 5

        for role_i, role in enumerate(['top', 'jgl', 'mid', 'adc', 'sup']):
            temp = 0.7
            most_likely_champ = ""
            champ_found = False
            red_crop = gray[role_dict[role][0]:role_dict[role][1], 1228:1262]

            for j, template in enumerate(red_champ_templates):
                champ_classify_percent = np.max(cv2.matchTemplate(
                    red_crop, template, cv2.TM_CCOEFF_NORMED))
                if(champ_classify_percent > temp):
                    champ_found = True
                    temp = champ_classify_percent
                    most_likely_champ = red_portraits[j][:-4]
            logger.info(f"Red {role} identified ({100*temp:.2f}%):  {most_likely_champ.capitalize()}")
            champs[role_i+5] = most_likely_champ
            identified += champ_found

        if identified < 10:
            for _ in range(frames_to_skip):
                cap.grab()
            _, frame = cap.read()
            logger.info("Too few champions found")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Grab portraits of each champion found, to search for on the minimap
    for champ_i, champ in enumerate(champs):
        champ_image = os.path.join(
            'assets',
            'tracking',
            'champ_tracking',
            f'{champ}.jpg')

        templates[champ_i] = cv2.imread(champ_image, 0)

    return templates, champs

def data_collect():
    directory = os.listdir("output")
    videos = []
    for video in directory:
        if os.path.exists("output/%s/positions.csv" % video):
            videos.append(video)

    roles = ['top', 'jgl', 'mid', 'adc', 'sup']*2

    # Read first dataframe
    df = pd.read_csv("output/%s/positions.csv" % videos[0])

    x = df['Seconds']

    # Use seconds as index and drop column
    df.index = df['Seconds']
    df.drop('Seconds', axis=1, inplace=True)
    df = df.T  # Transpose matrix

    # Remove duplicated seconds
    df = df.loc[:, ~df.columns.duplicated()]

    # Fill in missing seconds with NaN
    buffer = [i for i in range(1200) if i not in x.tolist()]
    for i in buffer:
        df[str(i)] = [(np.nan, np.nan)]*10

    # Sort columns by seconds
    df = df[df.columns[np.argsort(df.columns.astype(int))]]

    # Add context columns
    df['video'] = videos[0]
    df['champ'] = df.index
    df['roles'] = (roles)

    # Reorder columns
    y = df.columns[-3:].tolist()
    y.extend(df.columns[:-3].tolist())
    df = df[y]
    df.index = range(10)

    df.columns = df.columns.astype(str)

    # Repeat process for every game recorded and add to database
    for video in videos[1:]:
        df1 = pd.read_csv("output/%s/positions.csv" % video)
        x = df1['Seconds']

        df1.index = df1['Seconds']
        df1.drop('Seconds', axis=1, inplace=True)
        df1 = df1.T

        df1 = df1.loc[:, ~df1.columns.duplicated()]

        buffer = [i for i in range(1200) if i not in x.tolist()]
        for i in buffer:
            df1[str(i)] = [(np.nan, np.nan)]*10

        df1 = df1[df1.columns[np.argsort(df1.columns.astype(int))]]

        df1['video'] = video
        df1['champ'] = df1.index
        df1['roles'] = (roles)

        df1.columns = df1.columns.astype(str)

        y = df1.columns[-3:].tolist()
        y.extend(df1.columns[:-3].tolist())
        df1 = df1[y]
        df1.index = range(10)
        df = pd.concat([df, df1], ignore_index=True)

    # Replace '_' character from champ names
    # Caused by champ having multiple splash arts after a visual update
    df['champ'] = df['champ'].apply(lambda x: x.replace("_", ""))
    return(df)
