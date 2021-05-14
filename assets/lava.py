from assets.utils import get_header_borders, clean_for_directory 
from assets.constants import *

import cv2
import os
import numpy as np
import pandas as pd

from assets.graphing import GraphsOperator

from youtubesearchpython import Playlist, Video, StreamURLFetcher
from sympy.geometry import Point, Circle, intersection, Line
from sympy import N
from numpy.linalg import norm

class LAVA(GraphsOperator):
  def execute(self, url="", local=False, playlist=False, skip=0, minimap=False, graphs=False):
    df = self.gather_data(url, local, playlist, skip, minimap)

    if graphs is True:
      self.draw_graphs(df, df.video.iloc[0])

    print(f'Video {df.video.iloc[0]} complete')


  def gather_data(self, url="", local=False, playlist=False, skip=0, minimap=False):
    videos = self.parse_url(url, local, playlist, skip)
    full_data = pd.DataFrame()

    for video, url in videos.items():
      print(f'Running Video: {video}')
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
      print("Champs identified")

      # map borders
      map_coordinates = self.map_borders(self.cap, league='automatic')
      print(f"Map coordinates found: {','.join(map(str, map_coordinates))}")

      print("Tracking commenced")
      df = self.tracker(map_coordinates, champs)
      print("Tracking complete")

      df = self.interpolate(df)
      print("Interpolation completed")

      df['video'] = video

      df.to_csv(f'output/positions/{video}.csv', index=False)

      full_data = full_data.append(df)
      cv2.destroyAllWindows()

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
    threshold = 0.8
    count = 0

    while ret is True and header_found is False:
      count += 1

      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      cropped = gray[0:120, 500:700]

      for header in header_templates.keys():
          header_max_probability = np.max(
            cv2.matchTemplate(
              cropped, 
              header_templates[header], 
              cv2.TM_CCOEFF_NORMED))

          if header_max_probability > threshold:
              header_found = True
              header_temp = header
              threshold = header_max_probability

      self.cap.set(1, self.frames_to_skip*120*count)
      ret, frame = self.cap.read()

      if header_found is True:
        print(f'{header_temp} header collected after {count*120} seconds')
        self.header = header_templates[header_temp]
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

        roles = blue_champ_sidebar.keys()

        # Check the sidebar for each champion
        for role in roles:
          temp = 0.65
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

          print(f"Blue {role} identified ({100*temp:.2f}%):  {most_likely_champ.capitalize()}")
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

        for role in roles:
            temp = 0.65
            most_likely_champ = ""
            
            sidebar_borders = red_champ_sidebar[role]

            red_crop = gray[sidebar_borders]

            for j, template in enumerate(red_champ_templates):
                champ_classify_percent = np.max(cv2.matchTemplate(
                    red_crop, template, cv2.TM_CCOEFF_NORMED))

                if champ_classify_percent > temp:
                    temp = champ_classify_percent
                    most_likely_champ = red_portraits[j][:-4]

            print(f"Red {role} identified ({100*temp:.2f}%):  {most_likely_champ.capitalize()}")
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
            tracking_folder,
            'champ_tracking',
            f'{champ}.jpg')
        champs[side][role]['template'] = cv2.imread(champ_image, 0)

    return champs

  def map_borders(self, cap, league):
    inhib_templates = []

    if league == 'automatic':
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
      
      return (
        slice(
          height//2 + int(border2[1]),
          height//2 + int(border1[1])),
        slice(
          4*width//5 + int(border1[0]),
          4*width//5 + int(border2[0]))
      )
    else:
      borders = leagues[league]

      return (
        slice(borders[0], borders[1]),
        slice(borders[2], borders[3])
      )
    
    

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
          cropped_timer = gray[timer_borders]
          digits = []

          for num in digit_templates.keys():
              template = digit_templates[num]
              check = cv2.matchTemplate(
                  cropped_timer, template, cv2.TM_CCOEFF_NORMED)
              digit = np.where(check > 0.8)
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
                  point = (np.nan, np.nan)
                  pass

              data_entries.append({
                  'champ': champs[side][role]['champ'],
                  'role': role, 
                  'side': side, 
                  'coords': np.array([(point[0] + 7)/H, (point[1] + 7)/W]), 
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
  def interpolate(df_input):
    df_input['iter'] = np.repeat(range(df_input.shape[0]//10), 10)

    df = df_input.pivot(columns='champ', values='coords', index='iter')
    
    if isinstance(df.iloc[0, 0], str):
      df = df.applymap(lambda a : np.fromstring(a[1:-1], dtype=float, sep=' '))
    
    H=1
    W=1
    RADIUS = 0.4
    cols = df.columns
    for index,column in enumerate(cols):
      cols_team = list(cols)

      cols_team.remove(column)
      col = df[column]
      col = np.array(col)
      colt = np.concatenate(col)

      # If no points found, usually caused by a bug in champion identification
      if(np.all(np.all(np.isnan(colt)))): 
        df[column] = [(np.nan,np.nan)]*len(col)
      else:
        col_temp = col
        i = 0

        # Search through points until an actual location is found
        while(np.all(np.isnan(col[i]))):
          i += 1

        # If there are missing values at the start
        if(np.all(np.isnan(col[0]))):
          try: # Need to fix
            temp = 20
            found = False

            # Check every champion on the same team to see if any were near the first known location
            for col_team in cols_team:
              for n in range(5): #4 seconds either side
                check = norm(df[col_team][i-n] - col[i])
                if(check < temp):
                  temp = check
                  found = True
                  champ_found = col_team
                check = norm(df[col_team][i+n] - col[i])
                if(check < temp):
                  temp = check
                  found = True
                  champ_found = col_team
            # If an ally was found near the first known location
            if(found):
              # Assume the two walked together
              col_temp = pd.concat([df[champ_found][:i],(col[i:])])
          except:
            pass

        j = len(col)-1

        # Same thing for missing values at the end
        while(np.all(np.isnan(col[j]))):
          j -= 1
        if(np.all(np.isnan(col[len(col)-1]))):
          try:
            temp = 20
            found = False
            for col_team in cols_team:
              for n in range(5):
                check = norm(df[col_team][j-n] - col[j])
                if(check < temp):
                  temp = check
                  found = True
                  champ_found = col_team
                check = norm(df[col_team][j+n] - col[j])
                if(check < temp):
                  temp = check
                  found = True
                  champ_found = col_team
            if(found):
              col_temp = pd.concat([(col_temp[:j+1]),(df[champ_found][j+1:])])
          except:
            pass

        count = 0
        k = i
        col_temp2 = col_temp

        # Deal with large chunks of missing values in the middle
        while(k < len(col_temp2)-1):
          k+=1
          if(np.all(np.isnan(col_temp[k]))):
            count += 1
          else:
            if(count > 5): # Missing for more than 5 seconds
              point = col_temp[k]
              if(index < 5): # Blue Side
                circle_x = 0
                circle_y = H
              else: # Red Side
                circle_x = W
                circle_y = 0
              # If first location after disappearing is in the base
              if(norm(np.array(point) - np.array([circle_x,circle_y])) < RADIUS):
                # Fill in with location just before disappearing (Assuming they died/recalled)
                col_temp2 = pd.concat([pd.Series(col_temp2[:k-count]),
                        pd.Series([col_temp2[k-count-1]]*count),
                        pd.Series(col_temp2[k:])], ignore_index = True)
              # Otherwise, check if there were any allies nearby before and after disappearing
              else:
                closest = 20
                found_closest = False

                # For every ally champion
                for col_team in cols_team:
                  temp = 20
                  found = False
                  for i in range(5):
                    try:
                      check = norm(np.array(point) - np.array(df[col_team][k+i]))
                      if(check < temp):
                        temp = check
                        found = True

                      check = norm(np.array(point) - np.array(df[col_team][k-i]))
                      if(check < temp):
                        temp = check
                        found = True
                    except:
                      pass

                  # If ally found nearby just before disappearing
                  if(found):
                    temp2 = 20
                    for i in range(5):
                      try:                      
                        check2 = norm(np.array(col_temp[k-count-1]) - np.array(df[col_team][k-count-1+i]))
                        if(check2 < temp2):
                          temp2 = check2
                          found_closest = True

                        check2 = norm(np.array(col_temp[k-count-1]) - np.array(df[col_team][k-count-1-i]))
                        if(check2 < temp2):
                          temp2 = check2
                          found_closest = True
                      except:
                        pass

                  # If ally found nearby before and after disappearing
                  if(found_closest):
                    # Choose ally who was closest on average
                    average = (temp + temp2) / 2
                    if(average < closest):
                      closest = average
                      champ_found = col_team

                # Assume the two walked together
                if(found_closest):
                  col_temp2 = pd.concat([pd.Series(col_temp2[:k-count]),
                          df[champ_found][k-count:k],
                          pd.Series(col_temp2[k:])],ignore_index = True)
            count = 0
        df[column] = col_temp2
    for col in df.columns: ###########
      df[col] = list(zip(*map(
        lambda l: l.interpolate().round(3),
        list(
          map(pd.Series, 
          zip(*df[col]))))))

    # df.to_csv('test2.csv')
    df_melted = pd.melt(df.reset_index(), id_vars='iter')

    df_merged = pd.merge(
      df_input, 
      df_melted, 
      on=['champ', 'iter']
    ).drop(
      ['coords', 'iter'], 
      axis=1)

    df_merged['x'] = df_merged['value'].apply(lambda x : x[0])
    df_merged['y'] = df_merged['value'].apply(lambda x : x[1])
    df_merged.drop('value', axis=1, inplace=True)

    return(df_merged)