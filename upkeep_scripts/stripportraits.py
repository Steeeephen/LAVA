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
import argparse
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
        frame_input = input("Skip how many frames?: (q if image shows all players at level 1) ")
        if(frame_input.lower() in ('q','quit')):
            break
        # Sets video to frame frames_to_skip. Change until you have a frame where desired champion is isolated
        cap.set(1, int(frame_input))
        ret, frame = cap.read()
        cv2.imshow("All players level 1?", frame)
        cv2.waitKey()

    ret, frame = cap.read()
    gray = frame

        # 5 different vertical coordinates for 5 different roles
    locations = [110,180,247,315,383]

    print("""
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
    """)

    while(True):
    # 1-5 blue side Top-Support, 6-10 red side Top-Support
        try:
            i = int(input("Number (q to exit):\n"))-1
        except:
            break
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
        cv2.imwrite('../assets/classify/%s/%s.jpg' % (col, name),cropped)

if __name__ == "__main__":
    main()