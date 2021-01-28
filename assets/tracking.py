import cv2
import pandas as pd
import numpy as np

from assets.constants import DIGIT_TEMPLATES, BARON_TEMPLATE, BARON, ROLE_DICT
# from assets.utils import timer


#-----------------------------

# Track locations

#-----------------------------

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

def tracker(champs, header, cap, templates, map_coordinates, frames_to_skip, collect):
	# points = {key:[] for key in champs}
	# seconds_timer = []
	
	_,frame = cap.read()	
	hheight,hwidth, _ = frame.shape
	hheight = hheight//15
	hwidth1 = 6*(hwidth//13)
	hwidth2 = 7*(hwidth//13)

	H = map_coordinates[1] - map_coordinates[0]
	W = map_coordinates[3] - map_coordinates[2]

	# point_i = [(0,0)]*10
	data_entries = []

	roles = list(ROLE_DICT.keys())

	count=1
	while(True):
		count+=1
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
			for image_i in DIGIT_TEMPLATES:
				res = cv2.matchTemplate(cropped_timer, DIGIT_TEMPLATES[image_i], cv2.TM_CCOEFF_NORMED)
				
				# Try to find each digit in the timer
				digit = np.where(res > 0.75)
				if(digit[0].any()): # If found
					seen = set() # Track horizontal locations
					inp = (list(zip(*digit)))
					outp = [(a, b) for a, b in inp if not (b in seen or seen.add(b) or seen.add(b+1) or seen.add(b-1))] # Only add digit if the horizontal position hasn't already been filled
					for out in outp:
						nums[out[1]] = image_i[:1] # Save digit
			timer_ordered = ""

			# Sort time
			for num in (sorted(nums)):
				timer_ordered = ''.join([timer_ordered, nums[num]])
			
			# Add time to list as seconds value
			# print(timer(timer_ordered, ""))
			second = timer(timer_ordered, "")
			# seconds_timer.append((timer(timer_ordered,"")))

			#-----------------------------

			# Tracking champions

			#-----------------------------

			# Crop to minimap and Baron Nashor icon
			cropped = gray[map_coordinates[0]: map_coordinates[1], map_coordinates[2] : map_coordinates[3]]
			cropped_4 = gray[BARON[0]- 4:BARON[1]+ 4, BARON[2]- 4:BARON[3]+ 4]
			
			# Check for the baron spawning
			buffcheck  = cv2.matchTemplate(cropped_4, BARON_TEMPLATE,  cv2.TM_CCOEFF_NORMED)
			buffs  = np.where(buffcheck > 0.9)
			
			# Stop when the baron spawns
			if(buffs[0].any()):
				break
			else: 
				for template_i, template in enumerate(templates):
					
					matched = cv2.matchTemplate(cropped,template,cv2.TM_CCOEFF_NORMED)
					location = (np.where(matched == max(0.8, np.max(matched))))
				
					# If champion found, save their location
					try:
						point = next(zip(*location[::-1]))
						# point_i[template_i] = point
						cv2.rectangle(cropped, point, (point[0] + 14, point[1] + 14), 255, 2)
					except:
						point = [np.nan,np.nan]
						pass

					side = 'blue' if template_i <= 4 else 'red'
					role = roles[template_i % 5]

					data_entries.append({'champ':champs[template_i], 'role': role, 'side':side, 'x':(point[0] + 7)/H, 'y':(point[1] + 7)/W, 'second':second})
					# temp = np.array([(point[0] + 7)/H, (point[1] + 7)/W])
					# points[champs[template_i]].append(temp)
				# df = pd.DataFrame(d)
				# df.to_csv("ok.csv")
				# cv2.imshow('minimap',cropped)
				# cv2.waitKey()
				# break
				if(not collect):
				# Show minimap with champions highlighted
					cv2.imshow('minimap',cropped)
					if cv2.waitKey(1) & 0xFF == ord('q'):
						break
				for _ in range(frames_to_skip):
					cap.grab()
	
	return pd.DataFrame(data_entries)