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
# Change this to get different leagues: 'uklc' and 'lec' supported so far
league = "lec"

# Change this url to get different videos
playlist_url = "https://www.youtube.com/playlist?list=PLQFWRIgi7fPSfHcBrLUqqOq96r_mqGR8c"

# Change this to skip the first n videos of the playlist
videos_to_skip = 15

```
to fit your usage

Output will be sent to the 'output' directory for each game in the playlist

## Notes

107 champions trackable; Ornn and Trundle are known problem children, will be fixed in due time

90 blue side champions identifiable, 88 red side

Currently optimised only for LEC/UKLC games, more to follow

Please note that this is an early build, any queries can be directed to  loltracker.program [at] gmail.com
