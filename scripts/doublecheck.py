"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Double checking the dimensions:


This script is for double checking that the dimensions line up with a video of your choice
Each league may have a different overlay size, so this is used to make sure it lines up nicely

--------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import cv2
import numpy as np
import os
import pafy
import youtube_dl

#-----------------------------

# Change this playlist to your own
playlist_url = "https://www.youtube.com/playlist?list=PLQFWRIgi7fPQkixYKTF3WkiyWwyP4BTzj"

# Change until you get a frame with desired champion isolated on the minimap
frames_to_skip = 8000

# Skip n videos in the playlist
videos_to_skip = 81

#-----------------------------

playlist = pafy.get_playlist(playlist_url)
videos = []
for i in (playlist['items']):
	videos.append(i['pafy'].videoid)

# Grabs first video from playlist
v = pafy.new(videos[videos_to_skip])
play = v.getbest(preftype="mp4")

# Opens video using OpenCV
cap = cv2.VideoCapture(play.url)

# Skip to frame frames_to_skip
cap.set(1, frames_to_skip)

ret, frame = cap.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Uncomment to check minimap against different known dimensions, only shows minimap if correct
# cropped = gray[538:705, 1098:1265] #pcs
# cropped = gray[538:705, 1098:1265] #lpl/lck
cropped = gray[554:707, 1112:1267] #lcs
# cropped = gray[563:712, 1108:1258] #lec
# cropped = gray[545:708, 1105:1270] #uklc

cv2.imshow("minimap", cropped)
cv2.waitKey()

# Check for sword icon in overlay, should be visible if correct
hheight,hwidth, _ = frame.shape
hheight = hheight//15
hwidth1 = 6*(hwidth//13)
hwidth2 = 7*(hwidth//13)

cropped = gray[0:hheight, hwidth1:hwidth2]
cv2.imshow("sword icon", cropped)
cv2.waitKey()

# Check baron timer, should be very clear in centre of image if correct
cropped = gray[23:50, 1210:1250] #timer
cv2.imshow("timer", cropped)
cv2.waitKey()

# Check blue- & redside sidebar, should cut the side off a little bit if correct
cropped = gray[110:403, 25:45]
cv2.imshow("blue side sidebar", cropped)
cv2.waitKey()

cropped = gray[110:403, 1235:1255]
cv2.imshow("red side sidebar", cropped)
cv2.waitKey()

# Clear windows
cv2.destroyAllWindows()


