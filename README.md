# LolTracker

A program for tracking League of Legends players and providing analytics from Youtube videos

***
## Usage

### YouTube

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

### Local

Add your 720p videos to the `input/` folder and run the program with `python loltracker.py -t local`

***
Currently optimised for 2020 versions of:

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

Though small changes to the broadcast may throw some off

Output will be sent to the 'output' directory for each game in the playlist. Takes at least 100 seconds per game on a Lenovo Thinkpad T430

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

## Installation

* Download or clone loltracker directory

* Install python and pip

* Navigate to folder with levelone.py

* Install dependencies
```
$ pip install -r requirements.txt
```
* Run program

```
$ python loltracker.py -t youtube
```
or
```
$ python loltracker.py -t local
```

* Alternatively, for more streamlined data collection

```
$ python data_collection.py
```

## Upkeep

To keep the program up to date, several scripts are included in the *upkeep_scripts* folder

#### Championtrackimage.py

For adding champion portraits to be tracked on the in-game minimap. If a new champion is added to the game/portrait updated, their portraits must be added to the directory

#### Createmapbase.py

For creating a map base should you need to support a new minimap. This needs to be done for any new leagues the program supports and acts as a reference for the graphs

#### Doublecheck.py

This is for making sure that the dimensions of a video line up with the program's dimensions. Used to double check that the program will be cropping images properly

#### Stripportraits.py

For stripping champion portraits from the sidebars. These portraits are used for identifying the champions at the start of the game. 

## Issues

The dimensions are *slightly* different for each league so the raw numbers in the csvs are not always directly comparable between different leagues. Any comparisons should be normalised to their respective league's dimensions and compared considering that.

Vods with a lot of noise beforehand can slow down the program, vods with highlights beforehand can sometimes break the program. Best input is a vod with only in-game footage, but avoiding pre-game highlights is the biggest thing as the program will see the highlight, assume the game started and begin gathering data.

Proximity graphs need to be smoothed