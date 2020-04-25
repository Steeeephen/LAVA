# LolTracker

A program for tracking League of Legends players and providing analytics from in-game video

## Installation

### Ubuntu 18.04:

Install python and pip

```
$ sudo apt-get update
$ sudo apt-get install python3.6
$ sudo apt install python3-pip
```
Navigate to folder with levelone.py

Install dependencies
```
$ pip install -r requirements.txt
```
Run program

```
$ python loltracker.py
```

### Windows

Install [python 3.7](https://docs.python.org/3/using/windows.html), pip should automatically install as well

Navigate to folder with levelone.py

Install dependencies
```
$ pip install -r requirements_windows.txt
```
Run program

```
$ python loltracker.py
```

## Usage

Change the lines
```
# Change this to get different leagues: 'uklc', 'slo', 'lfl', 'ncs', 'pgn', 'hpm', 'lcs', 'pcs', 'lpl', 'bl', 'lck', 'eum' and 'lec' supported so far
league = "lec"

# Change this url to get different videos
playlist_url = "https://www.youtube.com/playlist?list=PLQFWRIgi7fPSfHcBrLUqqOq96r_mqGR8c"

# Change this to skip the first n videos of the playlist
videos_to_skip = 0
```
to fit your usage

Output will be sent to the 'output' directory for each game in the playlist. Takes at least 100 seconds per game on my machine (Thinkpad T430)

## Output

### Level One:
An html for each champion on blue & red side, totalling 10. The file will contain interactive graphs of each player's in-game position until the 90 second mark. 
![Level One Example](/markdown_assets/levelone_example.png)

### Jungle Track:
An html for each jungler. The html file has choropleth maps showing the jungler's positions from 0-8, 8-14 and 14-20 minutes.
![Jungle Example](/markdown_assets/jungle_example.png)

### Mid Track:
Similar to jungle track, but for midlaners. Focuses on different regions of the map
![Midlane Example](/markdown_assets/midlane_example.png)

### Support Track:
Similar to mid track, but for supports.
![Support Example](/markdown_assets/support_example.png)

All plots can be downloaded as a png by opening the html and clicking the corresponding button on each graph

## Notes

125 champions trackable so far; Ornn, Thresh and Trundle are known problem children, will be fixed in due time. Yuumi is a disaster

101 blue side champions identifiable so far, 111 red side

Currently optimised for:

* EU Masters (Europe)
* LEC (Europe)
* LFL (France)
* UKLC (Ireland & the UK)
* SLO (Spain)
* NCS (Nordic)
* PGN (Italy)
* BL (Belgium)
* HPM (Czech Republic/Slovakia)
* LPL (China)
* LCK (Korea)
* LCS (North America)
* PCS (Pacific)

These leagues have been tested for the level one tracker, but the individual role trackers have not been *fully* tested just yet. Still some work to do on double checking these

The dimensions are *slightly* different for each league so the raw numbers in the csvs are not always directly comparable between different leagues. Vods with a lot of noise beforehand can slow down the program, vods with highlights beforehand can break the program. Best input is a vod with only in-game footage, but avoiding pre-game highlights is the biggest thing as the program will see the highlight, assume the game started and begin gathering data.

Replays desync the time as the program assumes the game was paused anytime the game isn't live. On the to-do list

Please note that this is an early build, any queries can be directed to  loltracker.program [at] gmail.com
