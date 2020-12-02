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
import os
import argparse

parser = argparse.ArgumentParser(description = "Creating Minimap Base")
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
        videos = pafy.new([url.split('v=')[1]]).getbest(preftype="mp4")
        cap = cv2.VideoCapture(play.url)

    x = ()
    league = input("Desired code for league: ")

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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print("---\nTry coordinates until only the minimap is shown\n---")
    count=0
    while(True):
        count+=1
        input0 = input("Y-coordinate upper: (q to continue) ")
        if(input0.lower() in ('q','quit')):
            break
        else:
            input1 = int(input("Y-coordinate lower: "))
            input2 = int(input("X-coordinate left: "))
            input3 = int(input("X-coordinate right: "))
        # Change these dimensions until you only have the desired champion in an ~14x14px image
        input0_int = int(input0)
        cropped = gray[input0_int:input1, input2:input3]

        # Check image is ok
        cv2.imshow("Attempt #%s" % count,cropped)
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    print("Creating Map Base..")
    # Takes the average of 40,000 frames, adjust if map image unclear
    for i in range(40000):
        try:
            _, frame = cap.read()
            cropped = frame[input0_int:input1, input2:input3]
            
            # Adds latest frame to a set, to save the colours for each pixel
            x = x + (np.array(cropped),)
        except:
            break

    # Stack the set, making an entry for each pixel
    sequence = np.stack(x, axis=3)

    # Get the median value of each pixel and save
    result = np.median(sequence, axis = 3).astype(np.uint8)
    os.mkdir("../assets/%s" % league)
    Image.fromarray(result).save("../assets/%s/%s.png" % (league,league))

if __name__ == "__main__":
    main()