

***
# LolTracker

A program for tracking League of Legends players and providing analytics from Youtube videos

***
## Installation

* Download or clone loltracker directory

* Install python and pip

* Navigate to folder with levelone.py

* Install dependencies
```
$ pip install -r requirements.txt
```
* Run program
***


## Usage

Run with parser arguments:

|Argument|Addition|Explanation
|---|---|---|
|-l|League|Code corresponding to the league of the videos|
|-n|Videos to skip| Number of videos to skip in the playlist| 
|-c|Collect| Streamlined data collection (No printing/progress reports)|

The program can be used with either Youtube videos or local videos

For local videos, add your 720p videos to the `input/` folder and run the program with `python loltracker.py -v`. 

For Youtube playlists, add your playlist url with `python loltracker.py -p <playlist_url>`

***
Currently optimised for certain 2020 versions of:

|Code|League	|
|---|---|
| lec 		| LEC (Europe) |
| eum 		| EU Masters (Europe) |
| lfl		| LFL (France) |
| uklc 		| UK LoL Championship (Ireland & the UK) (Pre Summer 2020)|
| slo 		| Superliga Orange (Spain)|
| ncs 		| Nordic Championship Series (Nordic)|
| pgn 		| PG Nationals (Italy)|
| bl 		| Belgian League (Belgium)|
| hpm 		| Hitpoint Masters (Czech Republic/Slovakia)|
| lpl 		| LPL (China)|
| lck 		| LCK (Korea) |
| lcs 		| LCS (North America) (Pre Summer 2020)|
| lcsnew	| LCS (North America) (Summer 2020 overlay)|
| pcs		| PCS (Pacific)|
| w20		| Worlds 2020|

Output will be sent to the 'output' directory for each game in the playlist. 

***

For example, to run the program on a local list of LCS videos:

`$ python loltracker.py -v -l lcs`

Or the streamlined version:

`$ python loltracker.py -v -c -l lcs`

Or to run it on a YouTube playlist of LEC vods, skipping the first 2:

`$ python loltracker.py -l lec -p https://www.youtube.com/playlist?list=PLTCk8PVh_Zwmfpm9bvFzV1UQfEoZDkD7s -n 2`

***
## Output

The raw data is saved as a csv named positions.csv

As well as this, the program will automatically draw up some basic graphs about each individual game:

### Level One:
An html for each champion on blue & red side, totalling 10. The file will contain interactive graphs of each player's in-game position until the 90 second mark. 

![Level One Example](/assets/markdown_assets/levelone_example.png)

### Jungle Track:
An html for each jungler. The html file has choropleth maps showing the jungler's positions from 0-8, 8-14 and 14-20 minutes.
![Jungle Example](/assets/markdown_assets/jungle_example.png)

### Mid Track:
Similar to jungle track, but for midlaners. Focuses on different regions of the map
![Midlane Example](/assets/markdown_assets/midlane_example.png)

### Support Track:
Similar to mid track, but for supports.
![Support Example](/assets/markdown_assets/support_example.png)

### Proximity Tracker:
Tracks player proximity for supports and junglers throughout the game. Example shows a support's proximity to their adc
<p align = "center">
	<img src = "/assets/markdown_assets/proximity_sample.png" width = 1200>
</p> 

All plots can be downloaded as a png by opening the html and clicking the corresponding button on each graph

## Notes

143 champion portraits trackable so far

124 blue side champions identifiable so far, 131 red side

## Upkeep

To keep the program up to date, several scripts are included in the *upkeep_scripts* folder

## Issues

The dimensions are *slightly* different for each league so the raw numbers in the csvs are not always directly comparable between different leagues. Any comparisons should be normalised to their respective league's dimensions and compared considering that.

Vods with a lot of noise beforehand can slow down the program, vods with highlights beforehand can sometimes break the program. Best input is a vod with only in-game footage, but avoiding pre-game highlights is the biggest thing as the program will see the highlight, assume the game started and begin gathering data.

Proximity graphs need to be smoothed

Testing on Windows is no longer an option for the immediate future

## To-do

Track items (big job)

Track events (big job)

Upkeep Scripts 2.0

Continue repo format improvements

Follow time from in-game timer instead of baron countdown

Update to support more leagues

Finish champion support