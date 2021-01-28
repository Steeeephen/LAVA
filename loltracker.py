"""

.__         .__   __                        __                 
|  |   ____ |  |_/  |_____________    ____ |  | __ ___________ 
|  |  /  _ \|  |\   __\_  __ \__  \ _/ ___\|  |/ // __ \_  __ \
|  |_(  <_> )  |_|  |  |  | \// __ \\  \___|    <\  ___/|  | \/
|____/\____/|____/__|  |__|  (____  /\___  >__|_ \\___  >__|   
                                  \/     \/     \/    \       

Project     : LolTracker 

Version     : 1.2.5

Author      : Stephen O' Farrell (stephen.ofarrell64@gmail.com)

Purpose     : Automatically track spatiotemporal data in league of Legends broadcast videos

"""


#-----------------------------

# Imports

#-----------------------------

import pandas as pd
import numpy as np 
import cv2
import youtube_dl
import pafy
import argparse
import os
import sys

from assets.interpolation import interpolate 
from assets.utils import identify, output_folders, data_collect, clean_for_directory, headers, map_borders
from assets.graphing import draw_graphs
from assets.tracking import tracker

def main(args):
  parser = argparse.ArgumentParser(description = "LolTracker")

  parser.add_argument('-v', '--video', action = 'store_true', default =  False, help = 'Use local videos instead of Youtube playlist')
  parser.add_argument('-u', '--url', type=str, default =  '', help = 'Link to single Youtube video')
  parser.add_argument('-c', '--collect', action='store_true', default = False, help = 'Streamline data collection process')
  # parser.add_argument('-l', '--league', type=str, default = 'lec', help = 'Choose the league, see README for documentation')
  parser.add_argument('-p', '--playlist', type=str, default = 'https://www.youtube.com/playlist?list=PLTCk8PVh_Zwmfpm9bvFzV1UQfEoZDkD7s', help = 'YouTube playlist')
  parser.add_argument('-n', '--videos_to_skip', type=int, default = 0, help = 'Number of Videos to skip')

  args = parser.parse_args(args)

  local = args.video
  collect = args.collect
  playlist_url = args.playlist
  # league = args.league
  url = args.url
  single_video = url != ''
  videos_to_skip = args.videos_to_skip

  rows_list = []
  
  # overlay_swap = league == "lcsnew"
  # overlay_swap_w20 = league == 'w20' # to be reformatted for better scalability

  # league, MAP_COORDINATES = change_league(league)

  # Get the height and width of the chosen league's map as all measurements will be relative to that  
  # H, W = cv2.imread('assets/%s/%s.png' % (league,league)).shape[:2]

  # # Scoreboard is only ever up during live footage, this will filter useless frames
  # if(overlay_swap):
  #   # For summer 2020 lcs overlay, which inexplicably changes halfway through the split :))`
  #   header = cv2.imread("assets/headers/lcsheader.jpg", 0)
  #   header2 = cv2.imread("assets/headers/lcsheader2.jpg", 0)
  # elif(overlay_swap_w20):
  #   header = cv2.imread('assets/headers/w20header.jpg',0)
  # else:
  #   header = cv2.imread("assets/headers/header.jpg", 0)
  
  # Iterate through each video in the playlist, grabbing their IDs
  if(not single_video):
    if(not local):
      playlist = pafy.get_playlist(playlist_url)
      videos = []
      for item_i in (playlist['items']):
        videos.append(item_i['pafy'].videoid)
    elif(local):
      videos = os.listdir('input')
      videos.remove('.gitkeep')
  else:
    videos = [url]

  # Skipping videos     
  videos = videos[videos_to_skip:]

  total_videos = len(videos)

  # Run on each video
  for i, video in enumerate(videos):
    # Get the video url using pafy
    if(not local):
      v = pafy.new(video)
      play = v.getbest(preftype="mp4")
      video = clean_for_directory(play.title)
      cap = cv2.VideoCapture(play.url)
    else:
      cap = cv2.VideoCapture("input/%s" % video)
      
    print("Game %d of %d: %s" % (i+1, total_videos, video))
    
    # Create output folders
    output_folders(video)

    # Skip one second each time
    frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS))

    try:
      cap.set(1, frames_to_skip*int(url.split('t=')[1]))
    except:
      pass

    seconds_timer = []

    # headers
    frame_height, frame_width, header, count = headers(cap, frames_to_skip, collect)
    
    cap.set(1, frames_to_skip*120*(count-2))

    # if(overlay_swap):
      # templates, champs, header = identify(cap, frames_to_skip, overlay_swap,  collect, header, header2)
    templates, champs = identify(cap, frames_to_skip, collect, header, frame_height, frame_width)
    # else:
      # templates, champs, header = identify(cap, frames_to_skip, overlay_swap,  collect, header)

    # map borders
    map_coordinates = map_borders(cap, frames_to_skip, header, frame_height, frame_width)

    df = tracker(champs, header, cap, templates, map_coordinates, frames_to_skip, collect)

    # df = interpolate(df, map_coordinates)

    # Use the seconds array to sync up the points with the ingame timer
    # seconds_timer = np.array(seconds_timer).astype(int)
    # seconds_timer = seconds_timer[~np.isnan(seconds_timer)]

    # Add seconds
    # df = pd.concat([df,pd.DataFrame({'Seconds':seconds_timer})], axis=1)
    
    # Remove the values that went wrong (9999 means the program's prediction was too low, a highly negative number means it was too high)
    df = df[(df.second < 1200) & (df.second > 0)].sort_values('second')
    df.to_csv("ok.csv")
    
    rows_list = draw_graphs(df, map_coordinates, video, collect, rows_list)

    # Output raw locations to a csv
    df.to_csv("output/%s/positions.csv" % video, index = False)

  pd.DataFrame(rows_list).to_csv("output/proximities.csv")
  
  if(not single_video):
    final_df = data_collect()
    final_df.to_csv("output/collected_data.csv")
    return(final_df)
  else:
    return(df)

if __name__ == "__main__":
  main(sys.argv[1:])