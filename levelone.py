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

time0 = time.time()

if(not(os.path.exists("output"))):
	os.mkdir("output")
	
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
							circle_x = 5
							circle_y = 150
						else: # Red Side
							circle_x = 165
							circle_y = 0
						# If first location after disappearing is in the base
						if(norm(np.array(point) - np.array([circle_x,circle_y])) < 68):
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

# Putting interactive graphs to a html file
def graph_html(colour):
	if colour == 'blue':
		d = 0
	else:
		d = 5
	html_writer = open('output/%s/%s/levelonegraphs.html' % (video,colour),'w')
	html_text = """
		<html>
			<body>
				{0}
			</body>
		</html>
		""".format(divs).format(*div[d:d+5])

	html_writer.write(html_text)
	html_writer.close()

# Buffs spawn at 90 seconds exactly, when they appear we use this to sync the time
buff  = cv2.imread("assets/blueside_blue.jpg" , 0)
buff2 = cv2.imread("assets/blueside_red.jpg", 0)
buff3 = cv2.imread("assets/redside_blue.jpg", 0)
buff4 = cv2.imread("assets/redside_red.jpg", 0)

# Scoreboard is only ever up during live footage, this will filter useless frames
header = cv2.imread("assets/header.jpg", 0)

# Change this url to get different videos
plurl = "https://www.youtube.com/watch?v=jBHX225EVy4&list=PLQFWRIgi7fPSfHcBrLUqqOq96r_mqGR8c"

# Iterate through each video in the playlist, grabbing their IDs
playlist = pafy.get_playlist(plurl)
videos = []
for i in (playlist['items']):
	videos.append(i['pafy'].videoid)

# Change this to skip videos
videos = videos[13:]

# Run on each video
for i, video in enumerate(videos):
	v = pafy.new(video)
	play = v.getbest(preftype="mp4")
	video = play.title
	
	print("Game: %s" % video)
	
	# Create output folder
	if not os.path.exists("output/%s" % video):
		os.makedirs("output/%s" % video)
		os.makedirs("output/%s/red" % video)
		os.makedirs("output/%s/blue" % video)

	# Stream video url through OpenCV
	cap = cv2.VideoCapture(play.url)

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
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cropped = gray[0:hheight, hwidth1:hwidth2]

		# Look for the scoreboard
		matched = cv2.matchTemplate(cropped, header, cv2.TM_CCOEFF_NORMED)
		location = np.where(matched > 0.8)
		
		# Break when scoreboard found
		if(location[0].any()):
			break

		# Skip one second
		for i in range(frames_to_skip):
			cap.grab()

	threshold = 0.7
	champs = []
	portraits = os.listdir("classify/blue")
	

	# Identify blue side champions
	while(len(champs) != 5):
		blue = gray[105:410, 17:52]
		champs = []
		
		# Check the sidebar for each champion
		for portrait in portraits:
			template = cv2.imread("classify/blue/%s" % portrait, 0)
			matched  = cv2.matchTemplate(blue, template, cv2.TM_CCOEFF_NORMED)
			location = np.where(matched > threshold)
			
			if(location[0].any()):
				champs.append(champdict[portrait[:-4]])
				print("Blue Champion Identified: %s" % portrait[:-4])
		
		# If too many champions found, check again with stricter threshold
		if(len(champs) > 5):
			threshold += 0.1
			print("-"*30)
		# If too few champions found, this is often due to an awkward frame transition. Skip a frame and try again
		elif(len(champs) < 5):
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
	while(len(champs) != 10):
		red = gray[105:410, 1228:1262]	
		champs = []
		champs.extend(champstemp)
		for portrait in portraits:
			template = cv2.imread("classify/red/%s" % portrait, 0)
			matched = cv2.matchTemplate(red, template, cv2.TM_CCOEFF_NORMED)
			location = np.where(matched > threshold)
			if(location[0].any()):
				champs.append(champdict[portrait[:-4]])
				print("Red Champion Identified: %s" % portrait[:-4])
		
		if(len(champs) < 10):
			for i in range(frames_to_skip):
				cap.grab()
			_, frame = cap.read()
			print("-"*30)
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		elif(len(champs) > 10):
			threshold += 0.1
			print("-"*30)

	# Grab portraits of each champion found, to search for on the minimap
	for i, champ in enumerate(champs):
		templates[i] = cv2.imread('champs/%s' % champ,0)
	points = {key:[] for key in champs}
	


	while(True):
		_, frame = cap.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		cropped = gray[0:hheight, hwidth1:hwidth2]
		
		# Again, only consider locations where the scoreboard shows
		matched = cv2.matchTemplate(cropped,header,cv2.TM_CCOEFF_NORMED)
		location = np.where(matched > 0.75)
		if(location[0].any()):

			# Each broadcast has a slightly different minimap size, output currently optimised for LEC but updates incoming
			cropped = gray[560:710, 1100:1270] #lec
			# cropped = gray[550:708, 1100:1270] #lcs
			#cropped = gray[535:707, 1100:1270] #lpl

			# Check for the buffs spawning
			buffcheck  = cv2.matchTemplate(cropped, buff,  cv2.TM_CCOEFF_NORMED)
			buffcheck2 = cv2.matchTemplate(cropped, buff2, cv2.TM_CCOEFF_NORMED)
			buffcheck3 = cv2.matchTemplate(cropped, buff3, cv2.TM_CCOEFF_NORMED)
			buffcheck4 = cv2.matchTemplate(cropped, buff4, cv2.TM_CCOEFF_NORMED)
			
			buffs  = np.where(buffcheck  > 0.9)
			buffs2 = np.where(buffcheck2 > 0.9)
			buffs3 = np.where(buffcheck3 > 0.9)
			buffs4 = np.where(buffcheck4 > 0.9)
			
			# Stop when the buffs do spawn
			if(buffs[0].any() or buffs2[0].any() or buffs3[0].any() or buffs4[0].any()):
				break
			else: # Otherwise, look for each champions and save their location
				for i, template in enumerate(templates):
					matched = cv2.matchTemplate(cropped,template,cv2.TM_CCOEFF_NORMED)
					location = np.where(matched >= 0.8)

					# Outline character on map and save point if found, return NaN if not
					try:
						point = next(zip(*location[::-1]))
						point_i[i] = point
						cv2.rectangle(cropped, point, (point[0] + 14, point[1] + 14), 255, 2)
					except:
						point = (np.nan,np.nan)
						pass
					temp = np.array([point[0] + 7, point[1] + 7])
					points[champs[i]].append(temp)

				# Show minimap with all champions outlined
				cv2.imshow('minimap',cropped)
				if cv2.waitKey(1) & 0xFF == ord('q'):
					break
				for i in range(frames_to_skip):
					cap.grab()
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
			lambda l: l.replace(7,np.nan).interpolate().round(),
			list(
				map(pd.Series, 
				zip(*df[col]))))))

	# Add a seconds column, counting back from buffs spawning
	df['Seconds'] = pd.Series(range(90-len(points[champs[0]]),91))

	# Output locations to a csv
	df.to_csv("output/%s/levelonepositions.csv" % video, index = False)
	
	# Array for the graphs to go into
	cols = df.columns[:-1]
	div = [""]*len(cols)

	# Graph each champion's locations
	for i, col in enumerate(cols):
		champ = df[col]
		axes = list(zip(*champ))

		fig = px.scatter(
			df,
			x = axes[0], 
			y = axes[1],
			range_x = [0,170],
			range_y = [150, 0],
			width = 800,
			height = 800,
			hover_name = 'Seconds',
			color = 'Seconds',
			color_continuous_scale = "Oranges")

		fig.add_layout_image(
				dict(
					source=Image.open("assets/result_720p.png"),
					xref="x",
					yref="y",
					x=0,
					y=0,
					sizex=170,
					sizey=150,
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
			title = col.capitalize(),
			template = "plotly_white",
			xaxis_showgrid = False,
			yaxis_showgrid = False
			)
		
		fig.update_xaxes(showticklabels = False, title_text = "")
		fig.update_yaxes(showticklabels = False, title_text = "")
		div[i] = plotly.offline.plot(fig, output_type = 'div')

	divs = ('{{{:d}}}\n'*5).format(*range(5))

	# Output graphs to html
	graph_html('blue')
	graph_html('red')
	print("Level One Analysed\n")

time1 = time.time()
print("Runtime: %.2f seconds" % (time1-time0))