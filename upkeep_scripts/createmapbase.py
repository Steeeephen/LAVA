"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Creating the Map Base:


This script is for creating the minimap base, as used for the background of each graph
It works by saving the colours for each pixel of the map for thousands of frames and getting the average, this doesn't make the cleanest map but it will make a legible map with the exact dimensions for the target league

--------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np 
import cv2
from PIL import Image
import pafy
import youtube_dl

#-----------------------------

# Change to a playlist including a video from your target league
playlist_url = "https://www.youtube.com/playlist?list=PLQFWRIgi7fPSfHcBrLUqqOq96r_mqGR8c"

# Change to desired league
league = "pcs"

# Change to a frame at the start of the game
frames_to_skip = 2000

#-----------------------------

playlist = pafy.get_playlist(playlist_url)
videos = []
for i in (playlist['items']):
	videos.append(i['pafy'].videoid)

# Grabs first video from this playlist
v = pafy.new(videos[0])
play = v.getbest(preftype="mp4")

# Opens video using OpenCV
cap = cv2.VideoCapture(play.url)
x = ()

# Set to the first frame that shows live footage from the game
cap.set(1, frames_to_skip)

# Takes the average of 40,000 frames, adjust if map image unclear
for i in range(40000):
	try:
		_, frame = cap.read()
		cropped = frame[563:712, 1108:1258] #replace with dimensions of target minimap---------------------------------------------------*
		
		# Adds latest frame to a set, to save the colours for each pixel
		x = x + (np.array(cropped),)
	except:
		break

# Stack the set, making an entry for each pixel
sequence = np.stack(x, axis=3)

# Get the median value of each pixel and save
result = np.median(sequence, axis = 3).astype(np.uint8)
Image.fromarray(result).save("%s.png" % league)