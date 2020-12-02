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
import argparse
import sys,inspect

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from assets.constants import LEAGUES 

#-----------------------------

parser = argparse.ArgumentParser(description = "Saving Images of Champions")

parser.add_argument('-v', '--video', action = 'store_true', default =  False, help = 'Use local videos instead of Youtube playlist')
parser.add_argument('-p', '--playlist', type=str, default = 'https://www.youtube.com/playlist?list=PLTCk8PVh_Zwmfpm9bvFzV1UQfEoZDkD7s', help = 'YouTube playlist')
parser.add_argument('-u', '--url', type=str, default =  '', help = 'Link to single Youtube video')
parser.add_argument('-n', '--videos_to_skip', type=int, default = 0, help = 'Number of Videos to skip')

args = parser.parse_args()

def main():
    local = args.video
    playlist_url = args.playlist
    url = args.url
    videos_to_skip = args.videos_to_skip

    if(url == ''):
        if(not local):
          playlist = pafy.get_playlist(playlist_url)
          videos = []
          for item_i in (playlist['items']):
            videos.append(item_i['pafy'].videoid)
            v = pafy.new(videos[videos_to_skip])
            play = v.getbest(preftype="mp4")
            cap = cv2.VideoCapture(play.url)
        elif(local):
            videos = os.listdir('../input')
            videos.remove('.gitkeep')
            video = videos[videos_to_skip]
            cap = cv2.VideoCapture("../input/%s" % video)   
    else:
        play = pafy.new(url.split('v=')[1]).getbest(preftype="mp4")
        cap = cv2.VideoCapture(play.url)

    while(True):
        frame_input = input("Skip how many frames?: (q to continue) ")
        if(frame_input.lower() in ('q','quit')):
            break
        # Sets video to frame frames_to_skip. Change until you have a frame where desired champion is isolated
        cap.set(1, int(frame_input))
        ret, frame = cap.read()
        cv2.imshow("Is this the start of the game?", frame)
        cv2.waitKey()
    cv2.destroyAllWindows()

    ret, frame = cap.read()
    gray = frame

    print("Click on the window and press any key to continue, find the one that fits best")
  
    for league in LEAGUES.keys():
        mmap0, mmap1, mmap2, mmap3 = LEAGUES[league]
        cropped = gray[mmap0:mmap1, mmap2:mmap3]
        cv2.imshow(league, cropped)
        cv2.waitKey()

    # Check for sword icon in overlay, should be visible if correct
    hheight,hwidth, _ = frame.shape
    hheight = hheight//15
    hwidth1 = 6*(hwidth//13)
    hwidth2 = 7*(hwidth//13)

    cropped = gray[0:hheight, hwidth1:hwidth2]
    cv2.imshow("sword icon", cropped)
    cv2.waitKey()
    cv2.destroyAllWindows() 

    blue_top = gray[108:133, 20:50]
    blue_jgl = gray[178:203, 20:50]
    blue_mid = gray[246:268, 20:50]
    blue_adc = gray[310:340, 20:50]
    blue_sup = gray[380:410, 20:50]

    cv2.imshow("top", blue_top)
    cv2.waitKey()
    cv2.imshow("jgl", blue_jgl)
    cv2.waitKey()
    cv2.imshow("mid", blue_mid)
    cv2.waitKey()
    cv2.imshow("adc", blue_adc)
    cv2.waitKey()
    cv2.imshow("sup", blue_sup)
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

if __name__ == "__main__":
    main()


