"""

.__         .__   __                        __                 
|  |   ____ |  |_/  |_____________    ____ |  | __ ___________ 
|  |  /  _ \|  |\   __\_  __ \__  \ _/ ___\|  |/ // __ \_  __ \
|  |_(  <_> )  |_|  |  |  | \// __ \\  \___|    <\  ___/|  | \/
|____/\____/|____/__|  |__|  (____  /\___  >__|_ \\___  >__|   
                                  \/     \/     \/    \       

Project     : LolTracker 

Version     : 1.1.0

Author      : Stephen O' Farrell (stephen.ofarrell64@gmail.com)

Purpose     : Automatically track spatiotemporal data in league of Legends broadcast videos

Usage       : Change lines 48-54 to your desired values

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

from assets.interpolation import interpolate 
from assets.utils import change_league, identify, output_folders, data_collect
from assets.graphing import draw_graphs
from assets.tracking import tracker

parser = argparse.ArgumentParser(description = "LolTracker")

parser.add_argument('-v', '--video', action = 'store_true', default =  False, help = 'Use local videos instead of Youtube playlist')
parser.add_argument('-u', '--url', type=str, default =  '', help = 'Link to single Youtube video')
parser.add_argument('-c', '--collect', action='store_true', default = False, help = 'Streamline data collection process')
parser.add_argument('-l', '--league', type=str, default = 'lec', help = 'Choose the league, see README for documentation')
parser.add_argument('-p', '--playlist', type=str, default = 'https://www.youtube.com/playlist?list=PLTCk8PVh_Zwmfpm9bvFzV1UQfEoZDkD7s', help = 'YouTube playlist')
parser.add_argument('-n', '--videos_to_skip', type=int, default = 0, help = 'Number of Videos to skip')

args = parser.parse_args()

def main():
  local = args.video
  collect = args.collect
  playlist_url = args.playlist
  league = args.league
  url = args.url
  videos_to_skip = args.videos_to_skip

  rows_list = []
  
  overlay_swap = league == "lcsnew"
  overlay_swap_w20 = league == 'w20' # to be reformatted for better scalability

  league, MAP_COORDINATES = change_league(league)

  # Get the height and width of the chosen league's map as all measurements will be relative to that  
  H, W = cv2.imread('assets/%s/%s.png' % (league,league)).shape[:2]

  # Scoreboard is only ever up during live footage, this will filter useless frames
  if(overlay_swap):
    # For new lcs overlay
    header = cv2.imread("assets/headers/lcsheader.jpg", 0)
    header2 = cv2.imread("assets/headers/lcsheader2.jpg", 0)
  elif(overlay_swap_w20):
    header = cv2.imread('assets/headers/w20header.jpg',0)
  else:
    header = cv2.imread("assets/headers/header.jpg", 0)

  # Iterate through each video in the playlist, grabbing their IDs
  if(url == ''):
    if(not local):
      playlist = pafy.get_playlist(playlist_url)
      videos = []
      for item_i in (playlist['items']):
        videos.append(item_i['pafy'].videoid)
    elif(local):
      videos = os.listdir('input')
      videos.remove('.gitkeep')
  else:
    videos = [url.split('v=')[1]]
  
  # Skipping videos     
  videos = videos[videos_to_skip:]

  total_videos = len(videos)

  # Run on each video
  for i, video in enumerate(videos):
    # Get the video url using pafy
    if(not local):
      v = pafy.new(video)
      play = v.getbest(preftype="mp4")
      video = play.title
      cap = cv2.VideoCapture(play.url)
    else:
      cap = cv2.VideoCapture("input/%s" % video)
      
    print("Game %d of %d: %s" % (i+1, total_videos, video))
    
    # Create output folders
    output_folders(video)

    # Skip one second each time
    frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS))

    if(overlay_swap):
      templates, champs = identify(cap, frames_to_skip, overlay_swap,  collect, header, header2)
    else:
      templates, champs = identify(cap, frames_to_skip, overlay_swap,  collect, header)

    df, seconds_timer = tracker(champs, header, cap, templates, MAP_COORDINATES, frames_to_skip, collect)
    
    df = interpolate(df, H, W)

    # Use the seconds array to sync up the points with the ingame timer
    seconds_timer = np.array(seconds_timer).astype(int)
    seconds_timer = seconds_timer[~np.isnan(seconds_timer)]

    # Add seconds
    df = pd.concat([df,pd.DataFrame({'Seconds':seconds_timer})], axis=1)
    
    # Remove the values that went wrong (9999 means the program's prediction was too low, a highly negative number means it was too high)
    df = df[(df.Seconds < 1200) & (df.Seconds > 0)].sort_values('Seconds')
    
    rows_list = draw_graphs(df, H, W, league, video, collect, rows_list)

    # Output raw locations to a csv
    df.to_csv("output/%s/positions.csv" % video, index = False)
  data_collect()
  pd.DataFrame(rows_list).to_csv("output/proximities.csv")

if __name__ == "__main__":
  main()