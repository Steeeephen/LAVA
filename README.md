# LolTracker

A program for tracking League of Legends players and providing analytics from Youtube videos

## Usage

Change the lines
```
# Change this to get different leagues: 'uklc', 'slo', 'lfl', 'ncs', 'pgn', 'hpm', 'lcs', 'lcsnew', 'pcs', 'lpl', 'bl', 'lck', 'eum' and 'lec' supported so far
league = "lec"

# Change this url to get different videos
playlist_url = "https://www.youtube.com/playlist?list=PLQFWRIgi7fPSfHcBrLUqqOq96r_mqGR8c"

# Change this to skip the first n videos of the playlist
videos_to_skip = 0
```
to fit your usage

Output will be sent to the 'output' directory for each game in the playlist. Takes at least 100 seconds per game on my machine (Thinkpad T430)

## Output

The raw data is saved as a csv, named positions.csv

As well as this, the program will automatically draw up some basic graphs about each individual game:

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

### Proximity Tracker:
Tracks player proximity for supports and junglers throughout the game. Example shows a support's proximity to their adc
<p align = "center">
	<img src = "/markdown_assets/proximity_sample.png" width = 1200>
</p> 

All plots can be downloaded as a png by opening the html and clicking the corresponding button on each graph

## Notes

140 champions trackable so far; Ornn, Thresh and Trundle are known problem children, will be fixed in due time. Yuumi is a disaster

110 blue side champions identifiable so far, 123 red side. Can check current progress [here](https://docs.google.com/spreadsheets/d/14pUWbDw32owzKmMUSGVbgytgAz0lY9U6FDAcrYU0Za4/edit?usp=sharing)

Currently optimised for:

* EU Masters (Europe) - eum
* LEC (Europe) - lec
* LFL (France) - lfl
* UK LoL Championship (Ireland & the UK) - uklc
* Superliga Orange (Spain) - slo
* Nordic Championship Series (Nordic) - ncs
* PG Nationals (Italy) - pgn
* Belgian League (Belgium) - bl
* Hitpoint Masters (Czech Republic/Slovakia) - hpm
* LPL (China) - lpl
* LCK (Korea) - lck
* LCS (North America) - lcs
* LCS (North America) (Summer 2020 overlay) - lcsnew
* PCS (Pacific) - pcs

## Installation

Install python and pip

Navigate to folder with levelone.py

Install dependencies
```
$ pip install -r requirements.txt
```
Run program

```
$ python loltracker.py
```

## Issues

The dimensions are *slightly* different for each league so the raw numbers in the csvs are not always directly comparable between different leagues. Any comparisons should be normalised to their respective league's dimensions and compared considering that.

Vods with a lot of noise beforehand can slow down the program, vods with highlights beforehand can sometimes break the program. Best input is a vod with only in-game footage, but avoiding pre-game highlights is the biggest thing as the program will see the highlight, assume the game started and begin gathering data.

Proximity graphs need to be smoothed