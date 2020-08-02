"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Save champion image for tracking:


This script is for saving the champion image that will be tracked on the minimap
You need to find a frame that has the desired champion isolated on the minimap for best results
When such a frame is found, adjust line 45 until you get an 8x8px image only including the champion and it will save it to the directory for tracking

--------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np 
import cv2
import os 
import pafy
import youtube_dl

#-----------------------------

# Change this playlist to your own
playlist_url = "https://www.youtube.com/playlist?list=PLTCk8PVh_ZwnVkz--EYb_Ctz_GYn2ASXs"

# Change to desired champion
champ = "zyraadsada"

# Change until you get a frame with desired champion isolated on the minimap
frames_to_skip = 14000

#-----------------------------

playlist = pafy.get_playlist(playlist_url)
videos = []
for i in (playlist['items']):
	videos.append(i['pafy'].videoid)

v = pafy.new(videos[0])
play = v.getbest(preftype="mp4")
cap = cv2.VideoCapture(play.url)

# Sets video to frame frames_to_skip. Change until you have a frame where desired champion is isolated
cap.set(1, frames_to_skip)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Change these dimensions until you only have the desired champion in an ~8x8px image
cropped = gray[246:268, 17:52] #--------------------------------------------------------------------------------------*

# Check image is ok
cv2.imshow("ok",cropped)
cv2.waitKey()

# Saves image in correct directory
cv2.imwrite('../champs/%s.jpg' % champ, cropped)
