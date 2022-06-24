***

# Upkeep Scripts 2.0

These are the scripts needed to maintain and update the project as new LoL/broadcast updates come out

The usage is simple enough, add these parameters when running the scripts

|Argument|Variable Name|Explanation|
|---|---|---|
|`-l`|`local`|Use local videos instead of Youtube|
|`-u`|`url`|Link to Youtube video or path to local video|

So for example, to use a local video:

```sh
$ python3 champion_track_image.py -l -u '~/Downloads/test.mp4'
```

Or a youtube VOD:

```sh
$ python3 strip_portraits.py -u 'https://www.youtube.com/watch?v=xnEbF-a0k80'
```

***

## `champion_track_image`

For adding champion portraits to be tracked on the in-game minimap. If a new champion is added to the game/portrait updated, their portraits must be added to the directory using this script

First you must find a frame at which the champion portrait is isolated on the minimap. Then you must give the name of the champ and the 4 coordinates that result in a 14x14px image of the champion. This is best done through trial-and-error and multiple iterations until the perfect match is found.


## DoubleCheck

This is for making sure that the dimensions of a video line up with the program's dimensions. Used to double check that the program will be cropping images properly

Usage is just like the previous two, all you have to do is check if the pictures line up as you would expect


## `strip_portraits`

For stripping champion portraits from the sidebars. These portraits are used for identifying the champions at the start of the game.

Once you have found a frame with all champions at level one, you must fill in their position's number and name and their portrait will be added to the db

Numbers: 
* 1  : Blue side toplaner
* 2  : Blue side jungler
* 3  : Blue side midlaner
* 4  : Blue side ADC
* 5  : Blue side support

* 6  : Red side toplaner
* 7  : Red side jungler
* 8  : Red side midlaner
* 9  : Red side ADC
* 10 : Red side support

***