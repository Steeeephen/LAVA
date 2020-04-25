import pandas as pd
import numpy as np 
from numpy.linalg import norm as norm
import cv2
import time
import os
import plotly.express as px
import plotly
from PIL import Image
import base64
import hashlib
from Crypto.Cipher import AES
from Crypto import Random
import youtube_dl
import pafy
from crypt import champdict, picdict

#-----------------------------

# Change this to get different leagues: 'uklc', 'slo', 'lfl', 'ncs', 'pgn', 'hpm', 'lcs', 'pcs', 'lpl', 'bl', 'lck', 'eum' and 'lec' supported so far
league = "eum"

# Change this url to get different videos
playlist_url = "https://www.youtube.com/playlist?list=PLcvpEVobSi9cMrj93URZ6J5TIOa6suaT8"

# Change this to skip the first n videos of the playlist
videos_to_skip = 5

#-----------------------------

# Checking the runtime
time0 = time.time()

# Each league has different dimensions for their broadcast overlay. The dimensions are stored here. The first four numbers correspond to the corners of the minimap while the subsequent 16 correspond
# to the redside red buff, blueside red buff, redside blue buff and blueside blue buff respectively
leagues = {
	"uklc":	[
	545, 708, 1105, 1270, 581, 591, 1180, 1190, 660, 670, 1187, 1197,
	627, 637, 1225, 1235, 616, 626, 1140, 1150],
	"lec":	[
	563, 712, 1108, 1258, 596, 606, 1175, 1185, 667, 677, 1181, 1191, 
	638, 648, 1217, 1227, 627, 637, 1140, 1150],
	"lck": [
	535, 705, 1095, 1267, 572, 582, 1172, 1182, 656, 666, 1180, 1190, 
	621, 631, 1222, 1232, 608, 618 ,1132, 1142],
	"lcs": [
	554, 707, 1112, 1267, 587, 597, 1183, 1193, 662, 672, 1190, 1200,
	630, 640, 1227, 1237, 619, 629, 1146, 1156],
	"pcs": [
	538, 705, 1098, 1265, 570, 580, 1173, 1183, 652, 662, 1181, 1191,
	618, 628, 1222, 1232, 606, 616, 1133, 1143]
}

# Some broadcasts have the same dimensions
if league == "lpl": league = "lck"
if league == "eum": league = "lcs"
if league in ["slo", 'lfl', 'ncs', 'pgn', 'bl', 'hpm']: league = "uklc"

map_0, map_1, map_2, map_3 = leagues[league][:4]
buff_list = leagues[league][4:]
baron = [23, 50, 1207, 1230]

# 8 minutes, 14 minutes and 20 minutes are important breakpoints in the game, we'll split our data into those time intervals
timesplits = {480:"0-8", 840:"8-14", 1200:"14-20"}

# Get the height and width of the chosen league's map as all measurements will be relative to that	
h,w = cv2.imread('assets/%s/%s.png' % (league,league)).shape[:2]
radius = int(w/2.5)

# Filling in as many missing values as possible
def headfill(df):
	cols = df.columns
	for index,column in enumerate(cols):
		if(index < 5): # Blue side
			cols2 = list(cols)[:5]
		else: # Red side
			cols2 = list(cols)[5:]

		cols2.remove(column)
		col = df[column]
		col = np.array(col)
		colt = np.concatenate(col)
		
		# If no points found
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
					for col2 in cols2:
						for n in range(5): #4 seconds either side
							check = norm(df[col2][i-n] - col[i])
							if(check < temp):
								temp = check
								found = True
								champ_found = col2
							check = norm(df[col2][i+n] - col[i])
							if(check < temp):
								temp = check
								found = True
								champ_found = col2
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
					for col2 in cols2:
						for n in range(5):
							check = norm(df[col2][j-n] - col[j])
							if(check < temp):
								temp = check
								found = True
								champ_found = col2
							check = norm(df[col2][j+n] - col[j])
							if(check < temp):
								temp = check
								found = True
								champ_found = col2
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
							for col2 in cols2:
								temp = 20
								found = False
								for i in range(5):
									try:
										check = norm(np.array(point) - np.array(df[col2][k+i]))
										if(check < temp):
											temp = check
											found = True

										check = norm(np.array(point) - np.array(df[col2][k-i]))
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
											check2 = norm(np.array(x[k-count-1]) - np.array(df[col2][k-count-1+i]))
											if(check2 < temp2):
												temp2 = check2
												found_closest = True

											check2 = norm(np.array(x[k-count-1]) - np.array(df[col2][k-count-1-i]))
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
										champ_found = col2

							# Assume the two walked together
							if(found_closest):
								x2 = pd.concat([pd.Series(x2[:k-count]),
												df[champ_found][k-count:k],
												pd.Series(x2[k:])],ignore_index = True)
					count = 0
			df[column] = x2
	return(df)

# Each position has different regions that are ideal to focus on. These functions will classify which region each point is in
def classify_jgl(points):
	reds = [0]*9
	for point in points:
		if(norm(point - np.array([0,0]))   		< radius):
			reds[5]+=1
		elif(norm(point - np.array([149,0])) 	< radius):
			reds[6]+=1
		elif(norm(point - np.array([149,149])) 	< radius):
			reds[8]+=1
		elif(norm(point - np.array([0,149])) 	< radius):
			reds[7]+=1
		elif(point[0] < h - point[1] - h/10):
			if(point[0] < (4/5)*point[1]):
				reds[1]+=1
			else:
				reds[0]+=1
		elif(point[0] < h - point[1] + h/10):
			reds[4]+=1
		else:
			if(point[0] < (4/5)*point[1]):
				reds[2]+=1
			else:
				reds[3]+=1
	return(reds)

def classify_sup(points):
	reds = [0]*7
	for point in points:
		if(norm(point - np.array([149,0]))  < radius):
			reds[6]+=1
		elif(norm(point - np.array([0,149]))    < radius):
			reds[3]+=1
		elif(norm(point - np.array([149,149]))  < radius or point[0] > w-w/10 or point[1] > h-h/10):
			reds[5]+=1
		elif(point[0] < h - point[1] - h/10):
			reds[0]+=1
		elif(point[0] > h - point[1] + h/10):
			if(point[0] < point[1]):
				reds[2]+=1
			else:
				reds[1] += 1
		else:
			reds[4]+=1
	return(reds)

def classify_mid(points):
	reds = [0]*8
	for point in points:
		if(norm(point - np.array([149,0]))	< radius):
			reds[6]+=1
		elif(norm(point - np.array([0,149])) 	< radius):
			reds[7]+=1
		elif(norm(point - np.array([149,149])) 	< radius or point[0] > w-w/10 or point[1] > h-h/10):
			reds[5]+=1
		elif(norm(point - np.array([0,149])) 	< radius or point[0] < w/10 or point[1] < h/10):
			reds[4]+=1
		elif(point[0] < h - point[1] - h/10):
			reds[3]+=1
		elif(point[0] > h - point[1] + h/10):
			reds[2]+=1
		elif(point[0] < point[1]):
			reds[1]+=1
		elif(point[0] > point[1]):
			reds[0]+=1
	return(reds)

# Putting graphs to a html file
def graph_html(div_plot, colour, champ):
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

# Buffs spawn at 90 seconds exactly, when they appear we use this to sync the time
buff  = cv2.imread("assets/%s/blueside_blue_%s.jpg" % (league, league), 0)
buff2 = cv2.imread("assets/%s/blueside_red_%s.jpg" % (league, league), 0)
buff3 = cv2.imread("assets/%s/redside_blue_%s.jpg" % (league, league), 0)
buff4 = cv2.imread("assets/%s/redside_red_%s.jpg" % (league,league), 0)

# Baron spawns at 20mins, when it appears we use this to sync the time
baron_template = cv2.imread("assets/baron.jpg", 0)

# Scoreboard is only ever up during live footage, this will filter useless frames
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

# Run on each video
for i, video in enumerate(videos):

	# Get the video url using pafy
	v = pafy.new(video)
	play = v.getbest(preftype="mp4")
	video = play.title
	
	print("Game: %s" % video)
	
	# Create output folder
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

	# Stream video url through OpenCV
	cap = cv2.VideoCapture(play.url)

	# Templates stores the champion pictures to search for, while point_i saves the most recent point found for that champion
	templates = [0]*10
	point_i = [(0,0)]*10

	# Skip one second each time, reduce this for longer runtime but better accuracy
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
		matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
		location = np.where(matched > 0.75)
		
		# Break when scoreboard found
		if(location[0].any()):
			break

		# Skip one second if not
		for i in range(frames_to_skip):
			cap.grab()
	
	threshold = 0.7
	portraits = os.listdir("classify/blue")
	identified = 0	

	# Identify blue side champions
	while(identified != 5):
		identified = 0
		champs = [""]*10
		
		# Each champion has their own spot on the sidebar, we use that to identify which champions are in the game and which position they're in
		blue_top = gray[108:133, 17:52]
		blue_jgl = gray[178:203, 17:52]
		blue_mid = gray[245:275, 17:52]
		blue_adc = gray[312:342, 17:52]
		blue_sup = gray[380:410, 17:52]
		
		# Check the sidebar for each champion
		for portrait in portraits:
			template = cv2.imread("classify/blue/%s" % portrait, 0)
			for i,role in enumerate(['top','jgl','mid','adc','sup']):
				if(eval('np.where(cv2.matchTemplate(blue_%s, template, cv2.TM_CCOEFF_NORMED) > threshold)[0].any()' % (role))):
					print("Blue %s Identified: %s" % (role,portrait[:-4]))
					identified+=1
					champs[i] = champdict[portrait[:-4]]
		
		# If too many champions found, check again with stricter threshold
		if(identified > 5):
			threshold += 0.05
			print("-"*30)
		
		# If too few champions found, this is often due to an awkward frame transition. Skip a frame and try again
		elif(identified < 5):
			for i in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			print("-"*30)
	print("-"*30)

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
					print("Red %s Identified: %s" % (role,portrait[:-4]))
					identified+=1
					champs[i+5] = champdict[portrait[:-4]]

		if(identified < 10):
			for i in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			print("-"*30)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		elif(identified > 10):
			threshold += 0.05
			print("-"*30)

	# Grab portraits of each champion found, to search for on the minimap
	for i, champ in enumerate(champs):
		templates[i] = cv2.imread('champs/%s' % champ,0)

	# Every position will be stored
	points = {key:[] for key in champs}
	
	while(True):
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cropped = gray[0:hheight, hwidth1:hwidth2]
		
		# Again, only consider locations where the scoreboard shows
		matched = cv2.matchTemplate(cropped,header,cv2.TM_CCOEFF_NORMED)
		location = np.where(matched > 0.65)

		if(location[0].any()):
			# Crop to the minimap
			cropped = gray[map_0:map_1, map_2:map_3]

			# Search for each buff
			cropped_1 = gray[buff_list[12]-4:buff_list[13]+4, buff_list[14]-4:buff_list[15]+4]
			cropped_2 = gray[buff_list[4]- 4:buff_list[5]+ 4, buff_list[6]- 4:buff_list[7]+ 4]
			cropped_3 = gray[buff_list[8]- 4:buff_list[9]+ 4, buff_list[10]-4:buff_list[11]+4]
			cropped_4 = gray[buff_list[0]- 4:buff_list[1]+ 4, buff_list[2]- 4:buff_list[3]+ 4]
			
			# Check whether or not they spawned
			buffcheck  = cv2.matchTemplate(cropped_1, buff,  cv2.TM_CCOEFF_NORMED)
			buffcheck2 = cv2.matchTemplate(cropped_2, buff2, cv2.TM_CCOEFF_NORMED)
			buffcheck3 = cv2.matchTemplate(cropped_3, buff3, cv2.TM_CCOEFF_NORMED)
			buffcheck4 = cv2.matchTemplate(cropped_4, buff4, cv2.TM_CCOEFF_NORMED)
			
			buffs  = np.where(buffcheck  > 0.9)
			buffs2 = np.where(buffcheck2 > 0.9)
			buffs3 = np.where(buffcheck3 > 0.9)
			buffs4 = np.where(buffcheck4 > 0.9)
			
			# Stop when the buffs do spawn (90 seconds has been reached)
			if(buffs[0].any() or buffs2[0].any() or buffs3[0].any() or buffs4[0].any()):
				break
			else: # Until then, track each champion
				for i, template in enumerate(templates):
					matched = cv2.matchTemplate(cropped,template,cv2.TM_CCOEFF_NORMED)
					location = np.where(matched >= 0.8)

					# Outline character on map and save point if found, return NaN if not for interpolation
					try:
						point = next(zip(*location[::-1]))
						point_i[i] = point
						cv2.rectangle(cropped, point, (point[0] + 14, point[1] + 14), 255, 2)
					except:
						point = (np.nan,np.nan)
						pass

					# Each champion is a 14x14p image, and openCV returns the top-left corner. We add 7 to each dimension to make sure it's taking the centre of the champion
					temp = np.array([point[0] + 7, point[1] + 7])
					points[champs[i]].append(temp)

				# Show minimap with all champions outlined
				cv2.imshow('minimap',cropped)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				for i in range(frames_to_skip):
					cap.grab()
	
	# Cast points to a pandas DataFrame
	df = pd.DataFrame(points)

	# Rename columns to readable champion names
	collist= df.columns
	newcols = [picdict[i] for i in collist]
	df.columns = newcols

	# Fill in some missing values
	df = headfill(df)
	
	# Interpolate any unredeemable missing values
	for col in df.columns:
		df[col] = list(zip(*map(
			lambda l: l.interpolate().round(),
			list(
				map(pd.Series, 
				zip(*df[col]))))))

	level_one_points = len(points[champs[0]])
	
	# Add a seconds column, counting back from buffs spawning (90 seconds)
	df['Seconds'] = pd.Series(range(90-len(points[champs[0]]),91))

	# Output raw level one locations
	df.to_csv("output/%s/levelonepositions.csv" % video, index = False)
	
	# Ignore the seconds column when iterating through champions
	cols = df.columns[:-1]
	
	# We colour the maps differently depending on whether they are for blue or red side
	colour="blue"

	# Graph each champion's locations
	for i, col in enumerate(cols):
		if i > 4: # If we graphed all 5 blue side champions
			colour = "red"

		champ = df[col]
		axes = list(zip(*champ))

		fig = px.scatter(
			df,
			x = axes[0], 
			y = axes[1],
			range_x = [0,map_3 - map_2],
			range_y = [map_1 - map_0, 0],
			width = 800,
			height = 800,
			hover_name = 'Seconds',
			color = 'Seconds',
			color_continuous_scale = "Oranges")

		fig.add_layout_image(
				dict(
					source=Image.open("assets/%s/%s.png" % (league, league)),
					xref="x",
					yref="y",
					x=0,
					y=0,
					sizex = map_3 - map_2,
					sizey = map_1 - map_0, 
					sizing="stretch",
					opacity=1,
					layer="below")
		)

		fig.add_layout_image(
				dict(
					source=Image.open('portraits/%sSquare.png' % col.capitalize()),
					xref="x",
					yref="y",
					x=0,
					y=0,
					sizex=20,
					sizey=20,
					sizing="stretch",
					opacity=1,
					layer="below")
		)

		fig.update_layout(
			title = "%s: Level One" % col.capitalize(),
			template = "plotly_white",
			xaxis_showgrid = False,
			yaxis_showgrid = False
			)
		
		fig.update_xaxes(showticklabels = False, title_text = "")
		fig.update_yaxes(showticklabels = False, title_text = "")
		graph_html(plotly.offline.plot(fig, output_type = 'div'),colour,"levelone/%s" % col)

	print("Level One Analysed\n")
	
	# Continue past 90 seconds, track the players until the baron spawns (20 minutes/1200 seconds)
	while(True):
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cropped = gray[0:hheight, hwidth1:hwidth2]
		
		matched = cv2.matchTemplate(cropped,header,cv2.TM_CCOEFF_NORMED)
		location = np.where(matched > 0.65)
		if(location[0].any()):
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

				cv2.imshow('minimap',cropped)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				for i in range(frames_to_skip):
					cap.grab()
	
	df = pd.DataFrame(points)

	collist= df.columns
	newcols = [picdict[i] for i in collist]
	df.columns = newcols

	df = headfill(df)
	
	for col in df.columns:
		df[col] = list(zip(*map(
			lambda l: l.interpolate().round(),
			list(
				map(pd.Series, 
				zip(*df[col]))))))

	# Seconds column now goes back from 1200 seconds, rather than 90
	df['Seconds'] = pd.Series(range(1201-len(points[champs[0]]),1201))
	colour = "blue"
	
	for col in [df.columns[i] for i in [1,6]]:
		# We split these up into our important intervals
		for times in timesplits.keys():
			reds = classify_jgl(df[col][df['Seconds'] <= times])

			# We scale our points so that they fall nicely on [0,255], meaning easier to read graphs
			reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)), reds))
			fig = px.scatter(
					x = [], 
					y = [],
					range_x = [0,w],
					range_y = [h, 0],
					width = 800,
					height = 800)


			fig.update_layout(
					template = "plotly_white",
					xaxis_showgrid = False,
					yaxis_showgrid = False
					)

			fig.update_xaxes(showticklabels = False, title_text = "")
			fig.update_yaxes(showticklabels = False, title_text = "")

			# Different colours for each team
			fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"
			fig.update_layout(
				shapes=[
				dict(
						type="path",
						path = "M 0,0 L %d,%d L %d,0 Z" % (w/2,h/2,w),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),
					),
				dict(
						type="path",
						path = "M 0,0 L %d,%d L 0,%d Z" % (w/2,h/2,w),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),
					),
				
				dict(
						type="path",
						path = "M %d,%d L %d,%d L 0,%d Z" % (w,h,w/2, h/2,h),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),
					),
				dict(
						type="path",
						path = "M %d,%d L %d,%d L %d,0 Z" % (w,h,w/2,h/2,w),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),
					),
				dict(
						type="path",
						path = "M %d,%d L %d,%d L %d,0 L 0,%d Z" % (w/10,h, w,h/10,w-w/10,h-h/10),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
					),
					
				dict(
						type="circle",
						xref="x",
						yref="y",
						x0=-radius,
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
						y0=radius,
						x1=radius,
						y1=-radius,
						line_color="white",
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[6],reds[6]),
						x0=w-radius,
						y0=radius,
						x1=w+radius,
						y1=-radius,
						line_color="white",
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),
						x0=-radius,
						y0=h+radius,
						x1=radius,
						y1=h-radius,
						line_color="white",
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[8],reds[8]),
						x0=w-radius,
						y0=h+radius,
						x1=w+radius,
						y1=h-radius,
						line_color="white",
					)])
			fig.update_layout(
				title = "%s: %s" % (col.capitalize(), timesplits[times]),
				template = "plotly_white",
				xaxis_showgrid = False,
				yaxis_showgrid = False
				)
			graph_html(plotly.offline.plot(fig, output_type = 'div'), colour, "jungle/%s_%s" % (col, timesplits[times]))
		colour = "red"
	
	print("Junglers tracked")
	
	colour = "blue"
	
	for col in [df.columns[i] for i in [4,9]]:
		for times in timesplits.keys():
			reds = classify_sup(df[col][df['Seconds'] <= times])
			reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)-min(reds)), reds))
			
			fig = px.scatter(
					x = [], 
					y = [],
					range_x = [0,w],
					range_y = [h, 0],
					width = 800,
					height = 800)


			fig.update_layout(
					template = "plotly_white",
					xaxis_showgrid = False,
					yaxis_showgrid = False
					)

			fig.update_xaxes(showticklabels = False, title_text = "")
			fig.update_yaxes(showticklabels = False, title_text = "")
			fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"
			fig.update_layout(
				shapes=[
				    dict(
				            type="path",
				            path = "M 0,0 L %d,%d L %d,0 Z" % (w,h,w),
				            line=dict(
				                color='white',
				                width=2,
				            ),
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[1],reds[1]),
				    ),
				    dict(
				            type="path",
				            path = "M 0,0 L %d,%d L 0,%d Z" % (w,h,h),
				            line=dict(
				                color='white',
				                width=2,
				            ),
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[2],reds[2]),
				    ),
				    dict(
				            type="path",
				            path = "M 0,%d L %d,0 L 0,0 Z" % (h,w),
				            line=dict(
				                color='white',
				                width=2,
				            ),
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[0],reds[0]),
				    ),
				    dict(
				            type="rect",
				            x0=0,
				            y0=h,
				            x1=w,
				            y1=h-h/10,
				            line=dict(
				                color="white",
				                width=2,
				            ),
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
				        ),
				    dict(
				            type="rect",
				            x0=w-w/10,
				            y0=h,
				            x1=w,
				            y1=0,
				            line=dict(
				                color='white',
				                width=2,
				            ),
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
				        ),
				    dict(
				            type="circle",
				            xref="x",
				            yref="y",
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
				            x0=w-radius,
				            y0=h+radius,
				            x1=w+radius,
				            y1=h-radius,
				            line_color='white',
				        ),
				    dict( # Plotly doesn't accept the Arc SVG command so this is a workaround to make the lanes look smooth
				            type="rect",
				            x0=0,
				            y0=h,
				            x1=w,
				            y1=h-h/10+0.5,
				            line=dict(
				                color=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
				                width=2,
				            ),
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
				        ),
				    dict(
				            type="rect",
				            x0=w-w/10+0.5,
				            y0=h,
				            x1=w,
				            y1=0,
				            line=dict(
				                color=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
				                width=2,
				            ),
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
				        ),
				    dict(
				            type="path",
				            path = "M %d,%d L %d,%d L %d,0 L 0,%d Z" % (w/10,h, w,h/10,w-w/10,h-h/10),
				            line=dict(
				                color="white",
				                width=2,
				            ),
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[4],reds[4]),
				        ),
				    dict(
				            type="circle",
				            xref="x",
				            yref="y",
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[6],reds[6]),
				            x0=w-radius,
				            y0=radius,
				            x1=w+radius,
				            y1=-radius,
				            line_color='white',
				        ),
				    dict(
				            type="circle",
				            xref="x",
				            yref="y",
				            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[3],reds[3]),
				            x0=-radius,
				            y0=h+radius,
				            x1=radius,
				            y1=h-radius,
				            line_color="white",
				        )]) 
			fig.update_layout(
				title = "%s: %s" % (col.capitalize(), timesplits[times]),
				template = "plotly_white",
				xaxis_showgrid = False,
				yaxis_showgrid = False
				)
			graph_html(plotly.offline.plot(fig, output_type = 'div'), colour, "support/%s_%s" % (col,timesplits[times]))
		colour = "red"
	
	print("Supports tracked")

	colour = "blue"
	
	for col in [df.columns[i] for i in [2,7]]:
		fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"
		for times in timesplits.keys():
			reds = classify_mid(df[col][df['Seconds'] <= times])
			reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)-min(reds)), reds))
			
			fig = px.scatter(
					x = [], 
					y = [],
					range_x = [0,w],
					range_y = [h, 0],
					width = 800,
					height = 800)


			fig.update_layout(
					template = "plotly_white",
					xaxis_showgrid = False,
					yaxis_showgrid = False
					)

			fig.update_xaxes(showticklabels = False, title_text = "")
			fig.update_yaxes(showticklabels = False, title_text = "")

			fig.update_layout(
				shapes=[
				dict(
						type="path",
						path = "M 0,0 L %d,%d L %d,0 Z" % (w,h,w),
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),
					),
				dict(
						type="path",
						path = "M 0,0 L %d,%d L 0,%d Z" % (w,h,h),
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),
					),
				dict(
						type="path",
						path = "M %d,%d L %d,%d L %d,%d Z" % (w/10,h,w,h/10,w,h),
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),
					),
					
				dict(
						type="path",
						path = "M 0,%d L %d,%d L 0,0 Z" % (h-h/10,w-w/10,0),
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),
					),
				dict(
						type="rect",
						x0=0,
						y0=h,
						x1=w/10,
						y1=0,
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
					),
				dict(
						type="rect",
						x0=0,
						y0=h/10,
						x1=w,
						y1=0,
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						x0=-radius,
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
						y0=radius,
						x1=radius,
						y1=-radius,
						line_color='white',
					),
				dict(
						type="rect",
						x0=0,
						y0=h/10-0.5,
						x1=w,
						y1=0,
						line=dict(
							color=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
					),
				dict(
						type="rect",
						x0=0,
						y0=h,
						x1=w/10-0.5,
						y1=0,
						line=dict(
							color=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
					),
				dict(
						type="rect",
						x0=0,
						y0=h,
						x1=w,
						y1=h-h/10,
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
					),
				dict(
						type="rect",
						x0=w-w/10,
						y0=h,
						x1=w,
						y1=0,
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
						x0=w-radius,
						y0=h+radius,
						x1=w+radius,
						y1=h-radius,
						line_color='white',
					),
				dict(
						type="rect",
						x0=0,
						y0=h,
						x1=w,
						y1=h-h/10+0.5,
						line=dict(
							color=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
					),
				dict(
						type="rect",
						x0=w-w/10+0.5,
						y0=h,
						x1=w,
						y1=0,
						line=dict(
							color=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[6],reds[6]),
						x0=w-radius,
						y0=radius,
						x1=w+radius,
						y1=-radius,
						line_color='white',
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),
						x0=-radius,
						y0=h+radius,
						x1=radius,
						y1=h-radius,
						line_color="white",
					)
					
				])
			fig.update_layout(
				title = "%s: %s" % (col.capitalize(), timesplits[times]),
				template = "plotly_white",
				xaxis_showgrid = False,
				yaxis_showgrid = False
				)
			graph_html(plotly.offline.plot(fig, output_type = 'div'), colour, "mid/%s_%s" % (col,timesplits[times]))
		colour = "red"

	print("Mids tracked")
	
	# Output raw locations to a csv
	df.to_csv("output/%s/positions.csv" % video, index = False)
	
# Print final runtime
time1 = time.time()
print("Runtime: %.2f seconds" % (time1-time0))