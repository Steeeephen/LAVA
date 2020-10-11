import cv2
import numpy as np
import os 

from assets.constants import LEAGUES, ROLE_DICT, BLUE_PORTRAITS, RED_PORTRAITS, RED_CHAMP_TEMPLATES, BLUE_CHAMP_TEMPLATES

# Create output directory
if(not(os.path.exists("output"))):
	os.mkdir("output")

def change_league(league):
	if league in ["lpl"]: league = "lck"
	elif league in ["eum", 'lcsnew']: league = "lcs"
	elif league in ["slo", 'lfl', 'ncs', 'pgn', 'bl', 'hpm']: league = "uklc"


	# Dimensions for cropping the map
	MAP_0, MAP_1, MAP_2, MAP_3 = LEAGUES[league][:4]
	return league, [MAP_0, MAP_1, MAP_2, MAP_3]

def output_folders(video):
	if not os.path.exists("output/%s" % video):
		os.makedirs("output/%s" % video)
		os.makedirs("output/%s/red" % video)
		os.makedirs("output/%s/blue" % video)
		os.makedirs("output/%s/blue/jungle" % video)
		os.makedirs("output/%s/blue/levelone" % video)
		os.makedirs("output/%s/blue/support" % video)
		os.makedirs("output/%s/blue/mid" % video)
		os.makedirs("output/%s/red/levelone" % video)
		os.makedirs("output/%s/red/jungle" % video)
		os.makedirs("output/%s/red/support" % video)
		os.makedirs("output/%s/red/mid" % video)

	# Function to recursively clean up and convert the timer reading to seconds.
def timer(time_read, last):
	if(len(time_read) < 1):
		return(9999)
	if(len(time_read) == 1):
		timer_clean = last + time_read
		try:
			return(1200-(int(timer_clean[:-2])*60+int(timer_clean[-2:])))
		except:
			return(9999)
	elif(time_read[0] == '7' and time_read[1] == '7'):
		return(timer(time_read[2:], last+time_read[:1]))
	else:
		return(timer(time_read[1:], last + time_read[:1]))

# Putting graphs to a html file
def graph_html(div_plot, video, colour, champ):
	html_writer = open('output/%s/%s/%s.html' % (video, colour, champ),'w')
	html_text = """
		<html>
			<body>
				<center>{0}</center>
			</body>
		</html>
		""".format(div_plot)

	html_writer.write(html_text)
	html_writer.close()

def identify(cap, frames_to_skip, OVERLAY_SWAP, header, header2 = ""):
	
	_,frame = cap.read()	
	hheight,hwidth, _ = frame.shape
	hheight = hheight//15
	hwidth1 = 6*(hwidth//13)
	hwidth2 = 7*(hwidth//13)

	# Templates stores the champion pictures to search for, while point_i saves the most recent point found for that champion
	templates = [0]*10

	while(True):
		_,frame = cap.read()

		# Making the images gray will make template matching more efficient
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Search only in a section near the top, again for efficiency
		cropped = gray[0:hheight, hwidth1:hwidth2]
		
		# Look for the scoreboard
		if(OVERLAY_SWAP): # If new lcs overlay, we have to check for both possibilities
			matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
			location = np.where(matched > 0.75)
			if(location[0].any()):
				break

			matched = cv2.matchTemplate(cropped, header2, cv2.TM_CCOEFF_NORMED)
			location = np.where(matched > 0.75)
			if(location[0].any()): # If it's the second one
				header = header2  # Use the second one
				break
		else: # Otherwise, check for the sword icon normally
			matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
			location = np.where(matched > 0.75)
			if(location[0].any()):
				break

		# Skip one second if not
		for _ in range(frames_to_skip):
			cap.grab()


	# Grab portraits for identifying the champions played  #########cons
	identified = 0 	

	# Identify blue side champions until exactly 5 have been found
	while(identified != 5):
		identified = 0
		champs = [""]*10

		# Check the sidebar for each champion
		for role_i,role in enumerate(['top','jgl','mid','adc','sup']):
			temp = 0.7
			most_likely_champ = ""
			champ_found = False

			# Crop to each role's portrait in the video
			blue_crop = gray[ROLE_DICT[role][0]:ROLE_DICT[role][1], 20:50]
			
			# Scroll through each template and find the best match
			for j, template in enumerate(BLUE_CHAMP_TEMPLATES):
				champ_classify_percent = np.max(cv2.matchTemplate(blue_crop, template, cv2.TM_CCOEFF_NORMED))
				
				# If a better match than the previous best, log that champion
				if(champ_classify_percent > temp):
					champ_found = True
					temp = champ_classify_percent
					most_likely_champ = BLUE_PORTRAITS[j][:-4]

			print("Blue %s identified: %s (%.2f%%)" % (role, most_likely_champ, 100*temp))
			champs[role_i] = most_likely_champ
			identified += champ_found
		
		# If too few champions found, this is often due to an awkward frame transition. Skip a frame and try again
		if(identified < 5):
			for _ in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			print("-"*30)
	print("-"*30)
		
	# Same for red side champions
	while(identified != 10):
		identified = 5

		for role_i,role in enumerate(['top','jgl','mid','adc','sup']):
			temp = 0.7
			most_likely_champ = ""
			champ_found = False
			red_crop = gray[ROLE_DICT[role][0]:ROLE_DICT[role][1], 1228:1262]

			for j, template in enumerate(RED_CHAMP_TEMPLATES):
				champ_classify_percent = np.max(cv2.matchTemplate(red_crop, template, cv2.TM_CCOEFF_NORMED))
				if(champ_classify_percent > temp):
					champ_found = True
					temp = champ_classify_percent
					most_likely_champ = RED_PORTRAITS[j][:-4]
			print("Red %s identified: %s (%.2f%%)" % (role, most_likely_champ, 100*temp))
			champs[role_i+5] = most_likely_champ
			identified += champ_found

		if(identified < 10):
			for _ in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			print("-"*30)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
	# Grab portraits of each champion found, to search for on the minimap
	for champ_i, champ in enumerate(champs):
		templates[champ_i] = cv2.imread('assets/champs/%s.jpg' % champ,0)

	return templates, champs
