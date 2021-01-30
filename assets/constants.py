import cv2
import os

# Dimensions of in-game timer
BARON = [23, 50, 1207, 1230]

# Baron spawns at 20mins, when it appears we use this to sync the time
BARON_TEMPLATE = cv2.imread("assets/tracking/baron.jpg", 0)

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
if(os.path.exists("assets/tracking/images")):
	for image_temp in os.listdir("assets/tracking/images"):
		DIGIT_TEMPLATES[image_temp] = cv2.imread("assets/tracking/images/%s" % image_temp, 0)

	BLUE_PORTRAITS = os.listdir("assets/tracking/champ_classifying/blue")
	RED_PORTRAITS = os.listdir("assets/tracking/champ_classifying/red")	

	BLUE_CHAMP_TEMPLATES = [""]*len(BLUE_PORTRAITS)
	RED_CHAMP_TEMPLATES = [""]*len(RED_PORTRAITS)

	# Save templates for template matching
	for portrait_i, portrait in enumerate(BLUE_PORTRAITS):
		BLUE_CHAMP_TEMPLATES[portrait_i] = cv2.imread("assets/tracking/champ_classifying/blue/%s" % portrait, 0)
	for portrait_i, portrait in enumerate(RED_PORTRAITS):	
		RED_CHAMP_TEMPLATES[portrait_i] = cv2.imread("assets/tracking/champ_classifying/red/%s" % portrait, 0)