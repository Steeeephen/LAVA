# LolTracker

A program for tracking League of Legends players from in-game video

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
$ python levelone.py
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
$ python levelone.py
```


## Usage

Change the lines
```
# Change this to get different leagues: 'uklc', 'slo', 'lfl', 'lcs', 'pcs', 'lpl', 'lck' and 'lec' supported so far
league = "lec"

# Change this url to get different videos
playlist_url = "https://www.youtube.com/playlist?list=PLQFWRIgi7fPSfHcBrLUqqOq96r_mqGR8c"

# Change this to skip the first n videos of the playlist
videos_to_skip = 0
```
to fit your usage

Output will be sent to the 'output' directory for each game in the playlist

## Output

Three files will be output; a csv containing all player positions as well as a html for blue & red side. The html file will contain interactive graphs of each player's in-game position until the 90 second mark. These plots can be downloaded as a png by opening the html and clicking the button on each graph


## Notes

126 champions trackable; Ornn, Thresh and Trundle are known problem children, will be fixed in due time. Yuumi is a disaster

103 blue side champions identifiable, 109 red side

Currently optimised only for:

* LFL (France)
* UKLC (Ireland & the UK)
* SLO (Spain)
* LEC (Europe)
* LPL (China)
* LCK (Korea)
* LCS (North America)
* PCS (Pacific)


The dimensions are *slightly* different for each league so the raw numbers are not always directly comparable between different leagues. Vods with a lot of noise beforehand can slow down the program, vods with highlights beforehand can break the program. Best input is a vod with only in-game footage, but avoiding pre-game highlights is the biggest thing as the program will see the highlight, assume the game started and begin gathering data.


Please note that this is an early build, any queries can be directed to  loltracker.program [at] gmail.com
