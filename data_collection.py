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

# Change this to get different leagues: 'uklc', 'slo', 'lfl', 'ncs', 'pgn', 'hpm', 'lcs', 'lcsnew' 'pcs', 'lpl', 'bl', 'lck', 'eum' and 'lec' supported so far
league = "lcsnew"

# Change this url to get different videos
playlist_url = "https://www.youtube.com/playlist?list=PLQFWRIgi7fPQkixYKTF3WkiyWwyP4BTzj"

# Change this to skip the first n videos of the playlist
videos_to_skip = 80

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


# Some broadcasts have the same dimensions
if league in ["lpl"]: league = "lck"
if league in ["eum", 'lcsnew']: league = "lcs"
if league in ["slo", 'lfl', 'ncs', 'pgn', 'bl', 'hpm']: league = "uklc"

map_0, map_1, map_2, map_3 = leagues[league][:4]
baron = [23, 50, 1207, 1230]

# 8 minutes, 14 minutes and 20 minutes are important breakpoints in the game, we'll split our data into those time intervals
timesplits =  {480:"0-8", 840:"8-14", 1200:"14-20"}
timesplits2 = {480:0, 840:480, 1200:840}

# Get the height and width of the chosen league's map as all measurements will be relative to that	
h,w = cv2.imread('assets/%s/%s.png' % (league,league)).shape[:2]
radius = int(w/2.5)

# Filling in as many missing values as possible
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
			x = col
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
						x = pd.concat([df[champ_found][:i],(col[i:])])
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
						x = pd.concat([(x[:j+1]),(df[champ_found][j+1:])])
				except:
					pass

			count = 0
			k = i
			x2 = x

			# Deal with large chunks of missing values in the middle
			while(k < len(x2)-1):
				k+=1
				if(np.all(np.isnan(x[k]))):
					count += 1
				else:
					if(count > 5): # Missing for more than 5 seconds
						point = x[k]
						if(index < 5): # Blue Side
							circle_x = 0
							circle_y = h
						else: # Red Side
							circle_x = w
							circle_y = 0
						# If first location after disappearing is in the base
						if(norm(np.array(point) - np.array([circle_x,circle_y])) < radius):
							# Fill in with location just before disappearing (Assuming they died/recalled)
							x2 = pd.concat([pd.Series(x2[:k-count]),
											pd.Series([x2[k-count-1]]*count),
											pd.Series(x2[k:])], ignore_index = True)
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
											check2 = norm(np.array(x[k-count-1]) - np.array(df[col_team][k-count-1+i]))
											if(check2 < temp2):
												temp2 = check2
												found_closest = True

											check2 = norm(np.array(x[k-count-1]) - np.array(df[col_team][k-count-1-i]))
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
								x2 = pd.concat([pd.Series(x2[:k-count]),
												df[champ_found][k-count:k],
												pd.Series(x2[k:])],ignore_index = True)
					count = 0
			df[column] = x2
	return(df)

# Function to recursively clean up the timer reading. Often will confuse 17:74 for 177:74 or miss a few digits, this removes as many of them as possible and converts the reading to the value in seconds
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

# Baron spawns at 20mins, when it appears we use this to sync the time
baron_template = cv2.imread("assets/baron.jpg", 0)

# Scoreboard is only ever up during live footage, this will filter useless frames
if(overlay_swap):
	header = cv2.imread("assets/lcsheader.jpg", 0)
	header2 = cv2.imread("assets/lcsheader2.jpg", 0)
else:
	header = cv2.imread("assets/header.jpg", 0)

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
	
	# Create output folder
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

		if(overlay_swap):
			matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
			location = np.where(matched > 0.75)
			if(location[0].any()):
				break

			matched = cv2.matchTemplate(cropped, header2, cv2.TM_CCOEFF_NORMED)
			location = np.where(matched > 0.75)
			if(location[0].any()):
				header = header2
				break
		else:
			matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
			location = np.where(matched > 0.75)
			if(location[0].any()):
				break

		# Skip one second if not
		for i in range(frames_to_skip):
			cap.grab()
	
	threshold = 0.67
	portraits = os.listdir("classify/blue")
	identified = 0 	

	# Identify blue side champions
	while(identified != 5):
		identified = 0
		champs = [""]*10
		
		# Each champion has their own spot on the sidebar, we use that to identify which champions are in the game and which position they're in
		blue_top = gray[108:133, 24:46]
		blue_jgl = gray[178:203, 24:46]
		blue_mid = gray[246:268, 24:46]
		blue_adc = gray[314:336, 24:46]
		blue_sup = gray[382:404, 24:46]
		
		# Check the sidebar for each champion
		for portrait in portraits:
			template = cv2.imread("classify/blue/%s" % portrait, 0)
			for i,role in enumerate(['top','jgl','mid','adc','sup']):
				if(eval('np.where(cv2.matchTemplate(blue_%s, template, cv2.TM_CCOEFF_NORMED) > threshold)[0].any()' % (role))):
					if(champs[i] == ""):
						identified+=1
						champs[i] = portrait[:-4]
		
		# If too many champions found, check again with stricter threshold
		if(identified > 5):
			threshold += 0.05
		
		# If too few champions found, this is often due to an awkward frame transition. Skip a frame and try again
		elif(identified < 5):
			for i in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	champstemp = champs
	threshold = 0.7
	portraits = os.listdir("classify/red")
		
	# Same for red side champions
	while(identified != 10):
		identified = 5
		champs = champstemp
		
		red_top = gray[108:133, 1228:1262]
		red_jgl = gray[178:203, 1228:1262]
		red_mid = gray[245:275, 1228:1262]
		red_adc = gray[312:342, 1228:1262]
		red_sup = gray[380:410, 1228:1262]
		
		for portrait in portraits:
			template = cv2.imread("classify/red/%s" % portrait, 0)
			for i,role in enumerate(['top','jgl','mid','adc','sup']):
				if(eval('np.where(cv2.matchTemplate(red_%s, template, cv2.TM_CCOEFF_NORMED) > threshold)[0].any()' % (role))):
					identified+=1
					champs[i+5] = portrait[:-4]

		if(identified < 10):
			for i in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		elif(identified > 10):
			threshold += 0.05

	# Grab portraits of each champion found, to search for on the minimap
	for i, champ in enumerate(champs):
		templates[i] = cv2.imread('champs/%s.jpg' % champ,0)

	# Every position will be stored
	points = {key:[] for key in champs}
	seconds_timer = []
	while(True):
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cropped = gray[0:hheight, hwidth1:hwidth2]
		
		matched = cv2.matchTemplate(cropped,header,cv2.TM_CCOEFF_NORMED)
		location = np.where(matched > 0.65)
		if(location[0].any()):
			cropped_timer = gray[23:50, 1210:1250]
			
			nums = dict()
			for i in os.listdir("assets/images"):
				template = cv2.imread("assets/images/%s" % i, 0)
				res = cv2.matchTemplate(cropped_timer, template, cv2.TM_CCOEFF_NORMED)
				digit = np.where(res > 0.75)
				if(digit[0].any()):
					seen = set()
					inp = (list(zip(*digit)))
					outp = [(a, b) for a, b in inp if not (b in seen or seen.add(b) or seen.add(b+1) or seen.add(b-1))]
					for out in outp:
						nums[out[1]] = i[:1]
			timer_ordered	 = ""
			for i,num in enumerate(sorted(nums)):
				timer_ordered = ''.join([timer_ordered, nums[num]])
			
			seconds_timer.append((timer(timer_ordered,"")))

			cropped = gray[map_0:map_1, map_2:map_3]
			cropped_4 = gray[baron[0]- 4:baron[1]+ 4, baron[2]- 4:baron[3]+ 4]
			
			# Check for the baron spawning
			buffcheck  = cv2.matchTemplate(cropped_4, baron_template,  cv2.TM_CCOEFF_NORMED)
			buffs  = np.where(buffcheck  > 0.9)
			
			# Stop when the baron spawns
			if(buffs[0].any()):
				break
			else: 
				for i, template in enumerate(templates):
					matched = cv2.matchTemplate(cropped,template,cv2.TM_CCOEFF_NORMED)
					location = np.where(matched >= 0.8)

					try:
						point = next(zip(*location[::-1]))
						point_i[i] = point
						cv2.rectangle(cropped, point, (point[0] + 14, point[1] + 14), 255, 2)
					except:
						point = (np.nan,np.nan)
						pass

					temp = np.array([point[0] + 7, point[1] + 7])
					points[champs[i]].append(temp)
				for i in range(frames_to_skip):
					cap.grab()
	df = pd.DataFrame(points)

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

	# Output raw locations to a csv
	df.to_csv("output/%s/positions.csv" % video, index = False)
	print("Game %d of %d complete: %s" % (index+1, len(videos), video))
	
# Print final runtime
time1 = time.time()
print("Runtime: %.2f seconds" % (time1 - time0))