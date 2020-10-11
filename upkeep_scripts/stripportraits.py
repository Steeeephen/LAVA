"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Save champion image for classifying:


This script is for saving the sidebar champion image that will be used to identify which champions are in the game
You need to find a frame that has each champion at level one for best results
When such a frame is found, you can type the role + champion name into the command line to add them to the database

Numbers: 
1  : Blue side toplaner
2  : Blue side jungler
3  : Blue side midlaner
4  : Blue side ADC
5  : Blue side support

6  : Red side toplaner
7  : Red side jungler
8  : Red side midlaner
9  : Red side ADC
10 : Red side support

--------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import cv2
import numpy as np
import os
import pafy
import youtube_dl

#-----------------------------

# Change this playlist to your own
playlist_url = "https://www.youtube.com/playlist?list=PLVgS_BIOY01xJhrU7MLw2dNY-fuCx-SAB"

# Change until you get a frame with desired champion isolated on the minimap
frames_to_skip = 12000

# Skip n videos in the playlist
videos_to_skip = 22

#-----------------------------

playlist = pafy.get_playlist(playlist_url)
videos = []
for i in (playlist['items']):
	videos.append(i['pafy'].videoid)

# Grabs first video from playlist
v = pafy.new(videos[videos_to_skip])
play = v.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)

# Sets video to frame frames_to_skip. Change until you have a frame where desired champion is isolated
cap.set(1, frames_to_skip)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

print("Image should show live footage with every player at level one")
print("Press any key while on image to confirm")

# Double check to make sure the frame is at level one
cv2.imshow("check",gray)
cv2.waitKey() # 

# 5 different vertical coordinates for 5 different roles
locations = [110,180,247,315,383]

while(True):
	# 1-5 blue side Top-Support, 6-10 red side Top-Support
	i = int(input("Number:\n"))-1
	
	# Champion name
	name = input("Champ:\n")
	x = 25
	y = 45
	col = "blue"
	if(i >= 5): # If champ is on red side
		x = 1235
		y = 1255
		col = "red"

	# Crop image to portrait
	cropped = gray[locations[i % 5]:(locations[i % 5]+20), x:y]
	
	# Save image to directory
	cv2.imwrite('../classify/%s/%s.jpg' % (col, name),cropped)

	# Break with Ctrl+c or Ctrl+z