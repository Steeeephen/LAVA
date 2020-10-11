"""

.__         .__   __                        __                 
|  |   ____ |  |_/  |_____________    ____ |  | __ ___________ 
|  |  /  _ \|  |\   __\_  __ \__  \ _/ ___\|  |/ // __ \_  __ \
|  |_(  <_> )  |_|  |  |  | \// __ \\  \___|    <\  ___/|  | \/
|____/\____/|____/__|  |__|  (____  /\___  >__|_ \\___  >__|   
                                  \/     \/     \/    \       

Project 		: LolTracker 

Version 		: 1.1.0

Author 			: Stephen O' Farrell (stephen.ofarrell64@gmail.com)

Purpose 		: Automatically track spatiotemporal data in League of Legends broadcast videos

Usage 			: Change lines 48-54 to your desired values

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
from assets.utils import change_league, identify, output_folders
from assets.graphing import draw_graphs
from assets.tracking import tracker

parser = argparse.ArgumentParser(description = "LolTracker")

parser.add_argument('-t', '--video_type', type=str, default = 'youtube', help = 'youtube or local videos')
args = parser.parse_args()

video_type = args.video_type

def main():
	# Change this to get different leagues: 'uklc', 'slo', 'lfl', 'ncs', 'pgn', 'hpm', 'lcs', 'lcsnew' 'pcs', 'lpl', 'bl', 'lck', 'eum' and 'lec' supported so far. See README for documentation
	LEAGUE = "lec"

	# Change this url to get different videos
	PLAYLIST_URL = "https://www.youtube.com/playlist?list=PLTCk8PVh_Zwmfpm9bvFzV1UQfEoZDkD7s"

	# Change this to skip the first n videos of the playlist
	VIDEOS_TO_SKIP = 0

	# LCS Summer 2020 has a different overlay
	OVERLAY_SWAP = LEAGUE == "lcsnew"

	LEAGUE, MAP_COORDINATES = change_league(LEAGUE)

	# Get the height and width of the chosen league's map as all measurements will be relative to that	
	H, W = cv2.imread('assets/%s/%s.png' % (LEAGUE,LEAGUE)).shape[:2]

	# Scoreboard is only ever up during live footage, this will filter useless frames
	if(OVERLAY_SWAP):
		# For new lcs overlay
		header = cv2.imread("assets/lcsheader.jpg", 0)
		header2 = cv2.imread("assets/lcsheader2.jpg", 0)
	else:
		header = cv2.imread("assets/header.jpg", 0)

	# Iterate through each video in the playlist, grabbing their IDs
	if(video_type == 'youtube'):
		playlist = pafy.get_playlist(PLAYLIST_URL)
		videos = []
		for item_i in (playlist['items']):
			videos.append(item_i['pafy'].videoid)
	elif(video_type == 'local'):
		videos = os.listdir('input')

	# Skipping videos
	videos = videos[VIDEOS_TO_SKIP:]

	# Run on each video
	for video in videos:

		# Get the video url using pafy
		if(video_type == 'youtube'):
			v = pafy.new(video)
			play = v.getbest(preftype="mp4")
			video = play.title
			cap = cv2.VideoCapture(play.url)
		elif(video_type == 'local'):
			cap = cv2.VideoCapture("input/%s" % video)
			
		print("Game: %s" % video)
		
		# Create output folders
		output_folders(video)

		# Stream video url through OpenCV
		
		# cap = cv2.VideoCapture(video)

		# Skip one second each time
		frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS))

		templates, champs = identify(cap, frames_to_skip, OVERLAY_SWAP, header)

		df, seconds_timer = tracker(champs, header, cap, templates, MAP_COORDINATES, frames_to_skip)
		
		df = interpolate(df, H, W)

		# Use the seconds array to sync up the points with the ingame timer
		seconds_timer = np.array(seconds_timer).astype(int)
		seconds_timer = seconds_timer[~np.isnan(seconds_timer)]

		# Add seconds
		df = pd.concat([df,pd.DataFrame({'Seconds':seconds_timer})], axis=1)
		
		# Remove the values that went wrong (9999 means the program's prediction was too low, a highly negative number means it was too high)
		df = df[(df.Seconds < 1200) & (df.Seconds > 0)].sort_values('Seconds')
		
		draw_graphs(df, H, W, LEAGUE, video)

		# Output raw locations to a csv
		df.to_csv("output/%s/positions.csv" % video, index = False)

if __name__ == "__main__":
	main()