# LolTracker

A program for tracking League of Legends players from in-game video

## Installation

### Ubuntu 18.04:

Install python

```
$ sudo apt-get update
$ sudo apt-get install python3.6
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

## Usage

Change the line
```
plurl = "https://www.youtube.com/watch?v=jBHX225EVy4&list=PLQFWRIgi7fPSfHcBrLUqqOq96r_mqGR8c"
```
to include your own playlist

Output will be sent to the 'output' directory for each game in the playlist

## Notes

Installation on other machines has not been fully tested yet

107 champions trackable

90 blue side champions identifiable, 88 red side

Currently optimised only for LEC games, but dimensions for LPL/LCS minimap also in code. 
Note that the output file is based on the LEC minimap, meaning outputs from other leagues are slightly offset until more output maps added

Please note that this is an early build, any queries can be directed to stephen.ofarrell64 [at] gmail.com
