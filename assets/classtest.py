from assets.utils import get_header_borders, clean_for_directory 
from assets.constants import *

import cv2
import os
import numpy as np
import pandas as pd
import logging

from assets.graphing import GraphsOperator

from youtubesearchpython import Playlist, Video, StreamURLFetcher
from sympy.geometry import Point, Circle, intersection, Line
from sympy import N

logging.basicConfig(
  level=logging.INFO,
  filename='logs/logclass.log')
logger = logging.getLogger('gather')

class LolTracker(GraphsOperator):
  def gather_data(self, url="", local=False, playlist=False, skip=0, minimap=False):
    videos = self.parse_url(url, local, playlist, skip)
    logger.info("ERunning video")
    full_data = pd.DataFrame()

    for video, url in videos.items():
      self.cap = cv2.VideoCapture(url)
      self.frames_to_skip = int(self.cap.get(cv2.CAP_PROP_FPS))

      try:
        self.cap.set(1, self.frames_to_skip*int(self.url.split('t=')[1]))
      except:
        pass

      self.minimap = minimap

      # headers
      count = self.headers()

      if count is None:
        continue

      self.cap.set(1, self.frames_to_skip*120*(count-2))

      champs = self.identify(self.cap)
      logger.info("Champs identified")

      # map borders
      map_coordinates = self.map_borders(self.cap)
      logger.info(f"Map coordinates found: {','.join(map(str, map_coordinates))}")

      logger.info("Tracking commenced")
      df = self.tracker(map_coordinates, champs)
      logger.info("Tracking complete")

      df = self.interpolate(df)
      logger.info("Interpolation complete")

      df = df[df.second.notnull()].sort_values('second')

      df['video'] = video

      full_data = full_data.append(df)
      
    # Output raw locations to a csv
    return full_data

  def gather_info(self, url="", local=False, playlist=False, skip=0):
    videos = self.parse_url(url, local, playlist, skip)

    full_data = {}

    for video, url in videos.items():
      self.cap = cv2.VideoCapture(url)
      self.frames_to_skip = int(self.cap.get(cv2.CAP_PROP_FPS))

      try:
        self.cap.set(1, self.frames_to_skip*int(self.url.split('t=')[1]))
      except:
        pass
      
      # headers
      count = self.headers()

      if count is None:
        continue

      self.cap.set(1, self.frames_to_skip*120*(count-2))

      champs = self.identify(self.cap)

      for side in champs.keys():
        for role in champs[side].keys():
          champs[side][role] = champs[side][role]['champ']

      full_data[video] = champs
    return full_data

  def parse_url(self, url="", local=False, playlist=False, skip=0):
    videos = {}

    if local is False:
      fetcher = StreamURLFetcher()

      if playlist is True:
        playlist_data = Playlist.getVideos(url)
        videos_list = [x['link'] for x in playlist_data['videos']]
        videos_list = videos_list[skip:]
      else:
        if isinstance(url, list):
          videos_list = url
        else:
          videos_list = [url]
        
      for video_url in videos_list:
        video = Video.get(video_url)
        url = fetcher.get(video, 22)
          
        video = clean_for_directory(video['title'])

        videos[video] = url
    # If pulling from local
    else:
      if playlist is True:
        videos_list = os.listdir('input')
        videos_list.remove('.gitkeep')
        videos_list = videos_list[skip:]
      else:
        if isinstance(url, list):
          videos_list = url
        else:
          videos_list = [url]
        
      for video_file in videos_list:
        video_path = os.path.join(
          'input',
          video_file)

        video = os.path.splitext(video_file)[0]

        videos[video] = video_path
    
    return videos

  def headers(self):
    headers_path = os.path.join(
      tracking_folder,
      'headers')

    headers_list = os.listdir(headers_path)
    
    header_templates = {}
    for header in headers_list:
        header_directory = os.path.join(
            headers_path,
            header)

        header_templates[header] = cv2.imread(header_directory, 0)

    ret, frame = self.cap.read()
    header_borders = get_header_borders(frame.shape)

    header_found = False
    count = 0

    while ret is True and header_found is False:
      count += 1

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cropped = gray

      for header in header_templates.keys():
          matched = cv2.matchTemplate(cropped, header_templates[header], cv2.TM_CCOEFF_NORMED)
          location = np.where(matched > 0.8)

          if location[0].any():
              logger.info(f'Header Found: {header}')
              header_found = True
              break

      self.cap.set(1, self.frames_to_skip*120*count)
      ret, frame = self.cap.read()

      if header_found is True:
        logger.info(f'{header} header collected after {count*120} seconds')
        self.header = header_templates[header]
        return count

  def identify(self, cap):
    ret, frame = cap.read()
    header_borders = get_header_borders(frame.shape)

    roles = blue_champ_sidebar.keys()

    # Instantiate dictionary for champions
    champs = {'blue': {}, 'red': {}}

    for role in roles:
      champs['blue'][role] = {'champ': "", 'template': ""}
      champs['red'][role] = {'champ': "", 'template': ""}

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

    # Grab portraits for identifying the champions played
    identified = 0

    # Identify blue side champions until exactly 5 have been found
    while identified != 5:
        identified = 0

        roles = blue_champ_sidebar.keys()

        # Check the sidebar for each champion
        for role in roles:
          temp = 0.7
          most_likely_champ = ""
          
          sidebar_borders = blue_champ_sidebar[role]

          # Crop to each role's portrait in the video
          blue_crop = gray[sidebar_borders]

          # Scroll through each template and find the best match
          for j, template in enumerate(blue_champ_templates):
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
                  most_likely_champ = blue_portraits[j][:-4]

          logger.info(f"Blue {role} identified ({100*temp:.2f}%):  {most_likely_champ.capitalize()}")
          champs['blue'][role]['champ'] = most_likely_champ
          if most_likely_champ != "":
            identified += 1

        # If too few champions found, skip a frame and try again
        if identified < 5:
            for _ in range(self.frames_to_skip):
                cap.grab()

            _, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            logger.info("Too few champions found")

    # Same for red side champions
    while identified != 10:
        identified = 5

        for role in roles:
            temp = 0.7
            most_likely_champ = ""
            
            sidebar_borders = red_champ_sidebar[role]

            red_crop = gray[sidebar_borders]

            for j, template in enumerate(red_champ_templates):
                champ_classify_percent = np.max(cv2.matchTemplate(
                    red_crop, template, cv2.TM_CCOEFF_NORMED))

                if champ_classify_percent > temp:
                    temp = champ_classify_percent
                    most_likely_champ = red_portraits[j][:-4]

            logger.info(f"Red {role} identified ({100*temp:.2f}%):  {most_likely_champ.capitalize()}")
            champs['red'][role]['champ'] = most_likely_champ
            if most_likely_champ != "":
                identified += 1

        if identified < 10:
            for _ in range(self.frames_to_skip):
                cap.grab()
            
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            logger.info("Too few champions found")

    # Grab portraits of each champion found, to search for on the minimap
    for side in champs:
      for role in champs[side]:
        champ = champs[side][role]['champ']

        champ_image = os.path.join(
            tracking_folder,
            'champ_tracking',
            f'{champ}.jpg')
        champs[side][role]['template'] = cv2.imread(champ_image, 0)

    return champs

  def map_borders(self, cap):
    inhib_templates = []

    minimap_path = os.path.join(
      tracking_folder,
      "minimap")

    for i in range(4):
      inhib_path = os.path.join(
        tracking_folder,
        "minimap",
        f"inhib{i}.jpg")

      inhib_templates.append(cv2.imread(inhib_path, 0))

    inhibs_found = 0
    ret, frame = cap.read()
    height, width, _ = frame.shape

    while ret is True and inhibs_found < 4:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        crop_frame_corner = (
          slice(height//2, height),
          slice(4*width//5, width))

        cropped = gray[crop_frame_corner]

        inhib_locations = []
        inhibs_found = 0

        for inhib in inhib_templates:
            res = cv2.matchTemplate(cropped, inhib, cv2.TM_CCOEFF_NORMED)
            location = (np.where(res == max(0.65, np.max(res))))

            try:
                point = next(zip(*location[::-1]))
                inhibs_found += 1
            except:
                inhibs_found = 0
                logger.info("Not all inhibs found")
                for i in range(self.frames_to_skip):
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
    
    return [
      slice(
        height//2 + int(border2[1]),
        height//2 + int(border1[1])),
      slice(
        4*width//5 + int(border1[0]),
        4*width//5 + int(border2[0]))
    ]
    

  def tracker(self, map_coordinates, champs):
    H = map_coordinates[0].stop - map_coordinates[0].start
    W = map_coordinates[1].stop - map_coordinates[1].start

    data_entries = []

    ret, frame = self.cap.read()
    header_borders = get_header_borders(frame.shape)
        
    while ret is True:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cropped = gray[header_borders]

      matched = cv2.matchTemplate(cropped, self.header, cv2.TM_CCOEFF_NORMED)
      location = np.where(matched > 0.65)

      if location[0].any():
          cropped = gray[timer_borders]
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
          
          # Crop to minimap
          cropped = gray[map_coordinates]

          for side in champs.keys():
            for role in champs[side].keys():
              template = champs[side][role]['template']

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

              data_entries.append({
                  'champ': champs[side][role]['champ'],
                  'role': role, 
                  'side': side, 
                  'x': (point[0] + 7)/H,
                  'y': (point[1] + 7)/W, 
                  'second': second
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

  @staticmethod
  def interpolate(df):
    return df