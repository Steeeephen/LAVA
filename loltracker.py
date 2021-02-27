import pandas as pd
import numpy as np 
import cv2
import youtube_dl
import pafy
import argparse
import os
import sys

from assets.interpolation import interpolate 
from assets.utils import identify, data_collect, clean_for_directory, headers, map_borders
from assets.graphing import draw_graphs
from assets.tracking import tracker

# Create output directory
if(not(os.path.exists("output"))):
    os.mkdir("output")

def main(args):
  parser = argparse.ArgumentParser(description = "LolTracker")

  parser.add_argument('-v', '--video', action = 'store_true', default =  False, help = 'Use local videos instead of Youtube playlist')
  parser.add_argument('-u', '--url', type=str, default =  '', help = 'Link to single Youtube video or video filename')
  parser.add_argument('-c', '--collect', action='store_true', default = False, help = 'Streamline data collection process')
  parser.add_argument('-p', '--playlist', type=str, default = 'https://www.youtube.com/playlist?list=PLTCk8PVh_Zwmfpm9bvFzV1UQfEoZDkD7s', help = 'YouTube playlist')
  parser.add_argument('-n', '--videos_to_skip', type=int, default = 0, help = 'Number of Videos to skip')

  args = parser.parse_args(args)

  local = args.video
  collect = args.collect
  playlist_url = args.playlist
  url = args.url
  videos_to_skip = args.videos_to_skip

  single_video = url != ''

  rows_list = []
  
  # Iterate through each video in the playlist, grabbing their IDs
  if(not single_video is True):
    # If pulling vods from youtube
    if(not local is True):
      playlist = pafy.get_playlist(playlist_url)
      videos = []
      for item_i in (playlist['items']):
        videos.append(item_i['pafy'].videoid)
    # If pulling from local
    else:
      videos = os.listdir('input')
      videos.remove('.gitkeep')
  # If just one video
  else:
    videos = [url]

  # Skipping videos     
  videos = videos[videos_to_skip:]

  total_videos = len(videos)

  # Run on each video
  for i, video in enumerate(videos):
    # Get the video url using pafy
    if(not local is True):
      v = pafy.new(video)
      play = v.getbest(preftype="mp4")
      video = clean_for_directory(play.title)
      cap = cv2.VideoCapture(play.url)
    else:
      cap = cv2.VideoCapture("input/%s" % video)
      video = clean_for_directory(video.split('.')[0])
      
    print("Game %d of %d: %s" % (i+1, total_videos, video))

    positions_file = os.path.join(
      "output",
      video,
      "positions.csv")
    
    if not os.path.exists(positions_file) is True:
      os.makedirs("output/%s" % video)

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
      
      df = df[df.second.notnull()].sort_values('second')
      # df.to_csv("raw+da.csv")
      
      draw_graphs(df, video, collect)

      # Output raw locations to a csv
      df.to_csv(positions_file, index = False)
    else:
      print("Skipping, already ran")
  
  # if(not single_video is True):
  #   final_df = data_collect()
  #   final_df.to_csv("output/collected_data.csv")
  #   return(final_df)
  # else:
  #   return(df)

if __name__ == "__main__":
  main(sys.argv[1:])