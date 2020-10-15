"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Save champion image for tracking:


This script is for saving the champion image that will be tracked on the minimap
You need to find a frame that has the desired champion isolated on the minimap for best results
When such a frame is found, adjust line 45 until you get an 14x14px image only including the champion and it will save it to the directory for tracking

--------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import numpy as np 
import cv2
import os 
import pafy
import youtube_dl
import argparse

#-----------------------------

parser = argparse.ArgumentParser(description = "Saving Images of Champions")

parser.add_argument('-v', '--video', action = 'store_true', default =  False, help = 'Use local videos instead of Youtube playlist')
parser.add_argument('-p', '--playlist', type=str, default = 'https://www.youtube.com/playlist?list=PLTCk8PVh_Zwmfpm9bvFzV1UQfEoZDkD7s', help = 'YouTube playlist')
parser.add_argument('-n', '--videos_to_skip', type=int, default = 0, help = 'Number of Videos to skip')

args = parser.parse_args()

def main():
	local = args.video
	playlist_url = args.playlist
	videos_to_skip = args.videos_to_skip

	if(not local):
		playlist = pafy.get_playlist(playlist_url)
		videos = []
		for i in (playlist['items']):
			videos.append(i['pafy'].videoid)
		v = pafy.new(videos[videos_to_skip])
		play = v.getbest(preftype="mp4")
		cap = cv2.VideoCapture(play.url)
	else:
		videos = os.listdir("../input")
		videos.remove('.gitkeep')
		video = videos[videos_to_skip]
		cap = cv2.VideoCapture("../input/%s" % video)

	while(True):
		frame_input = input("Skip how many frames?: (q to continue) ")
		if(frame_input.lower() in ('q','quit')):
			break
		# Sets video to frame frames_to_skip. Change until you have a frame where desired champion is isolated
		cap.set(1, int(frame_input))
		ret, frame = cap.read()
		cv2.imshow("Is the champion isolated?", frame)
		cv2.waitKey()
	
	cv2.destroyAllWindows()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	champ = input("Champion Name: ")

	count=0
	while(True):
		count+=1
		input0 = input("Y-coordinate upper: (q to quit) ")
		if(input0.lower() in ('q','quit')):
			break
		else:
			input1 = int(input("Y-coordinate lower: "))
			input2 = int(input("X-coordinate left: "))
			input3 = int(input("X-coordinate right: "))
		# Change these dimensions until you only have the desired champion in an ~14x14px image
		cropped = gray[int(input0):input1, input2:input3]

		# Check image is ok
		cv2.imshow("Attempt #%s" % count,cropped)
		cv2.waitKey()
		cv2.destroyAllWindows()

	# Saves image in correct directory
	cv2.imwrite('../assets/champs/%s.jpg' % champ, cropped)

if __name__ == "__main__":
	main()