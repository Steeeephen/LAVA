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
  parser.add_argument('-p', '--playlist', type=str, default = 'https://www.youtube.com/playlist?list=PLTCk8PVh_Zwmfpm9bvFzV1UQfEoZDkD7s', help = 'YouTube playlist')
  parser.add_argument('-n', '--videos_to_skip', type=int, default = 0, help = 'Number of Videos to skip')

  args = parser.parse_args(args)

  local = args.video
  collect = args.collect
  playlist_url = args.playlist
  url = args.url
  single_video = url != ''
  videos_to_skip = args.videos_to_skip

  rows_list = []
  
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
      video = clean_for_directory(video.split('.')[0])
      
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

    templates, champs = identify(cap, frames_to_skip, collect, header, frame_height, frame_width)

    # map borders
    map_coordinates = map_borders(cap, frames_to_skip, header, frame_height, frame_width)

    df = tracker(champs, header, cap, templates, map_coordinates, frames_to_skip, collect)

    # df = interpolate(df, map_coordinates)
    
    # Remove the values that went wrong (9999 means the program's prediction was too low, a highly negative number means it was too high)
    df = df[(df.second < 1200) & (df.second > 0)].sort_values('second')
    df.to_csv("test_output.csv")
    
    draw_graphs(df, video, collect)

    # Output raw locations to a csv
    df.to_csv("output/%s/positions.csv" % video, index = False)
  
  if(not single_video):
    final_df = data_collect()
    final_df.to_csv("output/collected_data.csv")
    return(final_df)
  else:
    return(df)

if __name__ == "__main__":
  main(sys.argv[1:])