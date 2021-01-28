import cv2
import os

# # Minimap dimensions
LEAGUES = {
	"uklc":	[
	545, 708, 1105, 1270],
	"lec":	[
	563, 712, 1108, 1258],
	"lck": 	[
	535, 705, 1095, 1267],
	"lpl": 	[
	535, 705, 1095, 1267],
	"lcs": [
	554, 707, 1112, 1267],
	"pcs": [
	538, 705, 1098, 1265]
}


# Dimensions of in-game timer
BARON = [23, 50, 1207, 1230]

# 8 minutes, 14 minutes and 20 minutes are important breakpoints in the game, we'll split our data into those time intervals
TIMESPLITS =  {480:"0-8", 840:"8-14", 1200:"14-20"}
TIMESPLITS2 = {480:0, 840:480, 1200:840}

# Baron spawns at 20mins, when it appears we use this to sync the time
BARON_TEMPLATE = cv2.imread("assets/baron.jpg", 0)

# Dimensions of role portraits
ROLE_DICT = {
	"top":[108,133], 
	"jgl":[178,203], 
	"mid":[246,268],
	"adc":[310,340],
	"sup":[380,410]}

# The digits for reading the timer 
DIGIT_TEMPLATES = dict()

# won't be run if constants imported from another directory
if(os.path.exists("assets/images")):
	for image_temp in os.listdir("assets/images"):
		DIGIT_TEMPLATES[image_temp] = cv2.imread("assets/images/%s" % image_temp, 0)

	BLUE_PORTRAITS = os.listdir("assets/classify/blue")
	RED_PORTRAITS = os.listdir("assets/classify/red")	

	BLUE_CHAMP_TEMPLATES = [""]*len(BLUE_PORTRAITS)
	RED_CHAMP_TEMPLATES = [""]*len(RED_PORTRAITS)

	# Save templates for template matching
	for portrait_i, portrait in enumerate(BLUE_PORTRAITS):
		BLUE_CHAMP_TEMPLATES[portrait_i] = cv2.imread("assets/classify/blue/%s" % portrait, 0)
	for portrait_i, portrait in enumerate(RED_PORTRAITS):	
		RED_CHAMP_TEMPLATES[portrait_i] = cv2.imread("assets/classify/red/%s" % portrait, 0)