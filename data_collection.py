
#-----------------------------

# Imports

#-----------------------------

import pandas as pd
import numpy as np 
from numpy.linalg import norm as norm
import cv2
import time
import os
import plotly.express as px
import plotly
from PIL import Image
import youtube_dl
import pafy

#-----------------------------

# Input

#-----------------------------

# Change this to get different leagues: 'uklc', 'slo', 'lfl', 'ncs', 'pgn', 'hpm', 'lcs', 'lcsnew' 'pcs', 'lpl', 'bl', 'lck', 'eum' and 'lec' supported so far
league = "lcsnew"

# Change this url to get different videos
playlist_url = "https://www.youtube.com/playlist?list=PLQFWRIgi7fPQkixYKTF3WkiyWwyP4BTzj"

# Change this to skip the first n videos of the playlist
videos_to_skip = 80

#-----------------------------

# Variables

#-----------------------------

# Checking the runtime
time0 = time.time()

# Each league has different dimensions for their broadcast overlay. 
leagues = {
	"uklc":	[
	545, 708, 1105, 1270],
	"lec":	[
	563, 712, 1108, 1258],
	"lck": [
	535, 705, 1095, 1267],
	"lcs": [
	554, 707, 1112, 1267],
	"pcs": [
	538, 705, 1098, 1265]
}

# LCS Summer 2020 has a different overlay
overlay_swap = league == "lcsnew"

roles = ['top','jgl','mid','adc','sup']

# Each row of the proximities database
rows_list = []

# Dimensions of role portraits
role_dict = {
	"top":[108,133], 
	"jgl":[178,203], 
	"mid":[246,268],
	"adc":[310,340],
	"sup":[380,410]}

# Some broadcasts have the same dimensions
if league in ["lpl"]: league = "lck"
if league in ["eum", 'lcsnew']: league = "lcs"
if league in ["slo", 'lfl', 'ncs', 'pgn', 'bl', 'hpm']: league = "uklc"

# Dimensions for cropping the map
map_0, map_1, map_2, map_3 = leagues[league][:4]

# Dimensions of in-game timer
baron = [23, 50, 1207, 1230]

# 8 minutes, 14 minutes and 20 minutes are important breakpoints in the game, we'll split our data into those time intervals
timesplits =  {480:"0-8", 840:"8-14", 1200:"14-20"}
timesplits2 = {480:0, 840:480, 1200:840}

# Get the height and width of the chosen league's map as all measurements will be relative to that	
h,w = cv2.imread('assets/%s/%s.png' % (league,league)).shape[:2]
radius = int(w/2.5)

# Baron spawns at 20mins, when it appears we use this to sync the time
baron_template = cv2.imread("assets/baron.jpg", 0)

# Scoreboard is only ever up during live footage, this will filter useless frames
if(overlay_swap):
	# For new LCS overlay
	header = cv2.imread("assets/lcsheader.jpg", 0)
	header2 = cv2.imread("assets/lcsheader2.jpg", 0)
else:
	header = cv2.imread("assets/header.jpg", 0)


# The digits for reading the timer 
digit_templates = dict()

for image_temp in os.listdir("assets/images"):
	digit_templates[image_temp] = cv2.imread("assets/images/%s" % image_temp, 0)


#-----------------------------

# Functions

#-----------------------------

# Calculate proximities
def proximity(l, t, side, role, role1):
	plots = ""
	for i in l: # For each allied champion
		count = 0
		champ = df.columns[i]
		champ_to_check = pd.DataFrame((df[df.columns[t]]).apply(lambda x : np.array(x)))
		champ_teammate =pd.DataFrame((df[champ]).apply(lambda x : np.array(x)))

		diffs = [0]*len(champ_teammate)
		for j in range(len(champ_teammate)): # For every pair of points gathered
			try:	
				dist = norm(champ_to_check[df.columns[t]][j] - champ_teammate[champ][j]) # Get the distance between the two
				diffs[j] = dist

				# If within 50 units, they're considered 'close'
				if(dist < 50):
					count+=1
			except:
				pass
		rows_list.append({"video": video, "side":side, "role":role, "target":roles[i%5], "player":role1, "teammate":champs[i], "proximity":100*count/len(df['Seconds'])})

# Interpolation function
def headfill(df):
	cols = df.columns
	for index,column in enumerate(cols):
		if(index < 5): # Blue side
			cols_team = list(cols)[:5]
		else: # Red side
			cols_team = list(cols)[5:]

		cols_team.remove(column)
		col = df[column]
		col = np.array(col)
		colt = np.concatenate(col)
		
		# If no points found, usually caused by a bug in champion identification
		if(np.all(np.all(np.isnan(colt)))): 
			df[column] = [(np.nan,np.nan)]*len(col)
		else:
			col_temp = col
			i = 0

			# Search through points until an actual location is found
			while(np.all(np.isnan(col[i]))):
				i += 1

			# If there are missing values at the start
			if(np.all(np.isnan(col[0]))):
				try: # Need to fix
					temp = 20
					found = False
					
					# Check every champion on the same team to see if any were near the first known location
					for col_team in cols_team:
						for n in range(5): #4 seconds either side
							check = norm(df[col_team][i-n] - col[i])
							if(check < temp):
								temp = check
								found = True
								champ_found = col_team
							check = norm(df[col_team][i+n] - col[i])
							if(check < temp):
								temp = check
								found = True
								champ_found = col_team
					# If an ally was found near the first known location
					if(found):
						# Assume the two walked together
						col_temp = pd.concat([df[champ_found][:i],(col[i:])])
				except:
					pass

			j = len(col)-1

			# Same thing for missing values at the end
			while(np.all(np.isnan(col[j]))):
				j -= 1
			if(np.all(np.isnan(col[len(col)-1]))):
				try:
					temp = 20
					found = False
					for col_team in cols_team:
						for n in range(5):
							check = norm(df[col_team][j-n] - col[j])
							if(check < temp):
								temp = check
								found = True
								champ_found = col_team
							check = norm(df[col_team][j+n] - col[j])
							if(check < temp):
								temp = check
								found = True
								champ_found = col_team
					if(found):
						col_temp = pd.concat([(col_temp[:j+1]),(df[champ_found][j+1:])])
				except:
					pass

			count = 0
			k = i
			col_temp2 = col_temp

			# Deal with large chunks of missing values in the middle
			while(k < len(col_temp2)-1):
				k+=1
				if(np.all(np.isnan(col_temp[k]))):
					count += 1
				else:
					if(count > 5): # Missing for more than 5 seconds
						point = col_temp[k]
						if(index < 5): # Blue Side
							circle_x = 0
							circle_y = h
						else: # Red Side
							circle_x = w
							circle_y = 0
						# If first location after disappearing is in the base
						if(norm(np.array(point) - np.array([circle_x,circle_y])) < radius):
							# Fill in with location just before disappearing (Assuming they died/recalled)
							col_temp2 = pd.concat([pd.Series(col_temp2[:k-count]),
											pd.Series([col_temp2[k-count-1]]*count),
											pd.Series(col_temp2[k:])], ignore_index = True)
						# Otherwise, check if there were any allies nearby before and after disappearing
						else:
							closest = 20
							found_closest = False

							# For every ally champion
							for col_team in cols_team:
								temp = 20
								found = False
								for i in range(5):
									try:
										check = norm(np.array(point) - np.array(df[col_team][k+i]))
										if(check < temp):
											temp = check
											found = True

										check = norm(np.array(point) - np.array(df[col_team][k-i]))
										if(check < temp):
											temp = check
											found = True
									except:
										pass

								# If ally found nearby just before disappearing
								if(found):
									temp2 = 20
									for i in range(5):
										try:											
											check2 = norm(np.array(col_temp[k-count-1]) - np.array(df[col_team][k-count-1+i]))
											if(check2 < temp2):
												temp2 = check2
												found_closest = True

											check2 = norm(np.array(col_temp[k-count-1]) - np.array(df[col_team][k-count-1-i]))
											if(check2 < temp2):
												temp2 = check2
												found_closest = True
										except:
											pass

								# If ally found nearby before and after disappearing
								if(found_closest):
									# Choose ally who was closest on average
									average = (temp + temp2) / 2
									if(average < closest):
										closest = average
										champ_found = col_team

							# Assume the two walked together
							if(found_closest):
								col_temp2 = pd.concat([pd.Series(col_temp2[:k-count]),
												df[champ_found][k-count:k],
												pd.Series(col_temp2[k:])],ignore_index = True)
					count = 0
			df[column] = col_temp2
	return(df)

# Function to recursively clean up the and convert the timer reading to seconds
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
		return(timer(time_read[2:], last + time_read[:1]))
	else:
		return(timer(time_read[1:], last + time_read[:1]))

#-----------------------------

# Tracking

#-----------------------------

# Iterate through each video in the playlist, grabbing their IDs
playlist = pafy.get_playlist(playlist_url)
videos = []
for i in (playlist['items']):
	videos.append(i['pafy'].videoid)

# Skipping videos
videos = videos[videos_to_skip:]

# Create output directory
if(not(os.path.exists("output"))):
	os.mkdir("output")

print("Data Collection commenced")

# Run on each video
for index, video in enumerate(videos):

	# Get the video url using pafy
	v = pafy.new(video)
	play = v.getbest(preftype="mp4")
	video = play.title

	# Create output directory for csv file
	if not os.path.exists("output/%s" % video):
		os.makedirs("output/%s" % video)

	# Stream video url through OpenCV
	cap = cv2.VideoCapture(play.url)

	# Templates stores the champion pictures to search for, while point_i saves the most recent point found for that champion
	templates = [0]*10
	point_i = [(0,0)]*10

	# Skip one second each time, reduce this for longer runtime but better accuracy. The timer should still be synced up but hasn't been tested
	frames_to_skip = int(cap.get(cv2.CAP_PROP_FPS))

	_,frame = cap.read()
	hheight,hwidth, _ = frame.shape
	hheight = hheight//15
	hwidth1 = 6*(hwidth//13)
	hwidth2 = 7*(hwidth//13)

	# Search until in-game footage found
	while(True):
		_,frame = cap.read()

		# Making the images gray will make it more efficient
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# Search only in a section near the top, again for efficiency
		cropped = gray[0:hheight, hwidth1:hwidth2]

		# Look for the scoreboard
		if(overlay_swap): # If new lcs overlay, we have to check for both possibilities
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
	
	#-----------------------------

	# Identifying Champions

	#-----------------------------

	blue_portraits = os.listdir("classify/blue")
	red_portraits = os.listdir("classify/red")
	identified = 0 	

	blue_champ_templates = [""]*len(blue_portraits)
	red_champ_templates = [""]*len(red_portraits)

	# Save templates for template matching
	for portrait_i, portrait in enumerate(blue_portraits):
		blue_champ_templates[portrait_i] = cv2.imread("classify/blue/%s" % portrait, 0)
	for portrait_i, portrait in enumerate(red_portraits):	
		red_champ_templates[portrait_i] = cv2.imread("classify/red/%s" % portrait, 0)

	
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
			blue_crop = gray[role_dict[role][0]:role_dict[role][1], 20:50]
			
			# Scroll through each template and find the best match
			for j, template in enumerate(blue_champ_templates):
				champ_classify_percent = np.max(cv2.matchTemplate(blue_crop, template, cv2.TM_CCOEFF_NORMED))
				
				# If a better match than the previous best, log that champion
				if(champ_classify_percent > temp):
					champ_found = True
					temp = champ_classify_percent
					most_likely_champ = blue_portraits[j][:-4]

			champs[role_i] = most_likely_champ
			identified += champ_found
		
		# If too few champions found, this is often due to an awkward frame transition. Skip a frame and try again
		if(identified < 5):
			for _ in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			
	# Same for red side champions
	while(identified != 10):
		identified = 5

		for role_i,role in enumerate(['top','jgl','mid','adc','sup']):
			temp = 0.7
			most_likely_champ = ""
			champ_found = False
			red_crop = gray[role_dict[role][0]:role_dict[role][1], 1228:1262]

			for j, template in enumerate(red_champ_templates):
				champ_classify_percent = np.max(cv2.matchTemplate(red_crop, template, cv2.TM_CCOEFF_NORMED))
				if(champ_classify_percent > temp):
					champ_found = True
					temp = champ_classify_percent
					most_likely_champ = red_portraits[j][:-4]
			champs[role_i+5] = most_likely_champ
			identified += champ_found

		if(identified < 10):
			for _ in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		
	# Grab portraits of each champion found, to search for on the minimap
	for i, champ in enumerate(champs):
		templates[i] = cv2.imread('champs/%s.jpg' % champ,0)

	# Every position will be stored
	points = {key:[] for key in champs}
	seconds_timer = []

	#-----------------------------

	# Track locations

	#-----------------------------

	while(True):
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cropped = gray[0:hheight, hwidth1:hwidth2]
		
		matched = cv2.matchTemplate(cropped,header,cv2.TM_CCOEFF_NORMED)
		location = np.where(matched > 0.65)
		if(location[0].any()):
			cropped_timer = gray[23:50, 1210:1250]
			
			#-----------------------------

			# Timer 

			#-----------------------------

			nums = dict()

			# For each digit
			for image_i in digit_templates:
				res = cv2.matchTemplate(cropped_timer, digit_templates[image_i], cv2.TM_CCOEFF_NORMED)
				
				# Try to find each digit in the timer
				digit = np.where(res > 0.75)
				if(digit[0].any()): # If found
					seen = set() # Track horizontal locations
					inp = (list(zip(*digit)))
					outp = [(a, b) for a, b in inp if not (b in seen or seen.add(b) or seen.add(b+1) or seen.add(b-1))] # Only add digit if the horizontal position hasn't already been filled
					for out in outp:
						nums[out[1]] = image_i[:1] # Save digit
			timer_ordered	 = ""

			# Sort time
			for num in (sorted(nums)):
				timer_ordered = ''.join([timer_ordered, nums[num]])
			
			# Add time to list as seconds value
			seconds_timer.append((timer(timer_ordered,"")))

			#-----------------------------

			# Tracking champions

			#-----------------------------

			# Crop to minimap and Baron Nashor icon
			cropped = gray[map_0:map_1, map_2:map_3]
			cropped_4 = gray[baron[0]- 4:baron[1]+ 4, baron[2]- 4:baron[3]+ 4]
			
			# Check for the baron spawning
			buffcheck  = cv2.matchTemplate(cropped_4, baron_template,  cv2.TM_CCOEFF_NORMED)
			buffs  = np.where(buffcheck  > 0.9)
			
			# Stop when the baron spawns
			if(buffs[0].any()):
				break
			else: 
				for template_i, template in enumerate(templates):
					matched = cv2.matchTemplate(cropped,template,cv2.TM_CCOEFF_NORMED)
					location = np.where(matched >= 0.8)

					# If champion found, save their location
					try:
						point = next(zip(*location[::-1]))
						point_i[template_i] = point
						cv2.rectangle(cropped, point, (point[0] + 14, point[1] + 14), 255, 2)
					except:
						point = (np.nan,np.nan)
						pass

					temp = np.array([point[0] + 7, point[1] + 7])
					points[champs[template_i]].append(temp)

				# Show minimap with champions highlighted
				for _ in range(frames_to_skip):
					cap.grab()
	# Save data to Pandas dataframe
	df = pd.DataFrame(points)

	# Interpolate
	df = headfill(df)
	for col in df.columns:
		df[col] = list(zip(*map(
			lambda l: l.interpolate().round(),
			list(
				map(pd.Series, 
				zip(*df[col]))))))

	# Use the seconds array to sync up the points with the ingame timer
	seconds_timer = np.array(seconds_timer).astype(int)
	seconds_timer = seconds_timer[~np.isnan(seconds_timer)]

	df = pd.concat([df,pd.DataFrame({'Seconds':seconds_timer})], axis=1)
	
	# Remove the values that went wrong (9999 means the program's prediction was too low, a highly negative number means it was too high)
	df = df[df['Seconds'] < 1200]
	df = df[df['Seconds'] > 0].sort_values("Seconds")

	# Calculate 
	proximity([0,1,2,3], 4, "blue", "support", champs[4])
	proximity([0,2,3,4], 1, "blue", "jungle", champs[1])
	proximity([5,6,7,8], 9, "red", "support", champs[9])
	proximity([5,7,8,9], 6, "red", "jungle", champs[6])

	# Output raw locations to a csv
	df.to_csv("output/%s/positions.csv" % video, index = False)
	print("Game %d of %d complete: %s" % (index+1, len(videos), video))

	df2 = pd.DataFrame(rows_list)
	df2.to_csv("output/proximities.csv")

# Print final runtime
time1 = time.time()
print("Runtime: %.2f seconds" % (time1 - time0))