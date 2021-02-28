import pandas as pd
import numpy as np 
import cv2
import youtube_dl
import pafy
import argparse
import os
import sys
import datetime
import logging
import warnings

from assets.interpolation import interpolate 
from assets.utils import identify, data_collect, clean_for_directory, headers, map_borders
from assets.graphing import draw_graphs
from assets.tracking import tracker

warnings.filterwarnings('ignore')

def main(args):
  parser = argparse.ArgumentParser(description = "LolTracker")

  parser.add_argument('-v', '--video', action = 'store_true', default =  False, help = 'Use local videos instead of Youtube playlist')
  parser.add_argument('-u', '--url', type=str, default =  '', help = 'Link to single Youtube video or video filename')
  parser.add_argument('-m', '--minimap', action='store_true', default = False, help = 'Show ingame minimap')
  parser.add_argument('-g', '--graphs', action='store_true', default = False, help = 'Draw Graphs from data')
  parser.add_argument('-p', '--playlist', type=str, default = 'https://youtube.com/playlist?list=PLJIIsW8PQINACXncXzFm2mSByE6q8u3O8', help = 'YouTube playlist')
  parser.add_argument('-n', '--videos_to_skip', type=int, default = 0, help = 'Number of Videos to skip')

  args = parser.parse_args(args)

  local = args.video
  minimap = args.minimap
  playlist_url = args.playlist
  url = args.url
  videos_to_skip = args.videos_to_skip
  graphs=args.graphs

  single_video = url != ''

  overall_logger = logging.getLogger('overall')
  overall_logger.setLevel(logging.INFO)
  format_string = ("%(asctime)s — %(message)s")
  log_format = logging.Formatter(format_string)

  # Creating and adding the console handler
  for hdlr in overall_logger.handlers[:]:  # remove all old handlers
      overall_logger.removeHandler(hdlr)

  os.makedirs('logs', exist_ok=True)
  log_file = f'logs/app.log'
  file_handler = logging.FileHandler(log_file)
  file_handler.setFormatter(log_format)
  overall_logger.addHandler(file_handler)

  # Iterate through each video in the playlist, grabbing their IDs
  if single_video is False:
    # If pulling vods from youtube
    if local is False:
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

  total_vids = len(videos)

  # Run on each video
  for i, video in enumerate(videos):    
    # Get the video url using pafy
    if(local is False):
      v = pafy.new(video)
      play = v.getbest(preftype="mp4")
      video = clean_for_directory(play.title)
      cap = cv2.VideoCapture(play.url)
    else:
      cap = cv2.VideoCapture("input/%s" % video)
      video = clean_for_directory(video.split('.')[0])

    overall_logger.info("Game %d of %d: %s" % (i+1, total_vids, video))

    positions_file = os.path.join(
      "output",
      video,
      "positions.csv")
    
    if not os.path.exists(positions_file):
      os.makedirs(f"output/{video}", exist_ok=True)
      os.makedirs(f"logs/{video}", exist_ok=True)

      logger = logging.getLogger(__name__)
      logger.setLevel(logging.INFO)

      # Creating and adding the console handler
      for hdlr in logger.handlers[:]:  # remove all old handlers
          logger.removeHandler(hdlr)

      log_file = f'logs/{video}/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.log'
      overall_logger.info(f'Logging run to {log_file}')

      file_handler = logging.FileHandler(log_file)
      file_handler.setFormatter(log_format)
      logger.addHandler(file_handler)

      logger.info("Beginning run")

      # Skip one second each time
      frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS))
      logger.info(f"Number of frames between datapoints: {frames_to_skip}")
      
      try:
        cap.set(1, frames_to_skip*int(url.split('t=')[1]))
      except:
        pass

      seconds_timer = []

      # headers
      frame_height, frame_width, header, count = headers(cap, frames_to_skip, logger)
      logger.info(f'Header collected after {count*120} seconds')
      
      cap.set(1, frames_to_skip*120*(count-2))

      templates, champs = identify(cap, frames_to_skip, header, frame_height, frame_width, logger)
      logger.info("Champs identified")

      # map borders
      map_coordinates = map_borders(cap, frames_to_skip, header, frame_height, frame_width)
      logger.info(f"Map coordinates found: {','.join(map(str, map_coordinates))}")

      logger.info("Tracking commenced")
      df = tracker(champs, header, cap, templates, map_coordinates, frames_to_skip, minimap)
      logger.info("Tracking complete")

      # df = interpolate(df, map_coordinates)
      logger.info("Interpolation complete")

      df = df[df.second.notnull()].sort_values('second')
      
      if graphs is True:
        logger.info("Drawing graphs")
        draw_graphs(df, video, logger)
        logger.info("Graphs drawn")

      # Output raw locations to a csv
      df.to_csv(positions_file, index = False)
      logger.info(f"Data saved to {positions_file}")
    else:
      overall_logger.info(f"Skipping, {video} data already collected")
  
  # if(not single_video is True):
  #   final_df = data_collect()
  #   final_df.to_csv("output/collected_data.csv")
  #   return(final_df)
  # else:
  #   return(df)

if __name__ == "__main__":
  main(sys.argv[1:])