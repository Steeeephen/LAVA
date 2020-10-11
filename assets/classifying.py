import numpy as np

from numpy.linalg import norm

def classify_jgl(points, H, W, RADIUS):
	reds = [0]*9
	points.dropna(inplace=True)
	for point in points:
		if(norm(point - np.array([0,0]))   		< RADIUS): # Toplane
			reds[5]+=1
		elif(norm(point - np.array([149,0])) 	< RADIUS): # Red base
			reds[6]+=1
		elif(norm(point - np.array([149,149])) 	< RADIUS): # Botlane
			reds[8]+=1
		elif(norm(point - np.array([0,149])) 	< RADIUS): # Blue base
			reds[7]+=1
		elif(point[0] < H - point[1] - H/10): # Above midlane upper border
			if(point[0] < (5/4)*point[1]):
				reds[1]+=1 # Blue side
			else:
				reds[0]+=1 # Red side
		elif(point[0] < H - point[1] + H/10): # Above midlane lower border (in midlane)
			reds[4]+=1 
		elif(point[0] > H - point[1] + H/10): # Below midlane lower border
			if(point[0] < (5/4)*point[1]): # Blue side
				reds[2]+=1
			elif(point[0] > (5/4)*point[1]): # Red side
				reds[3]+=1
	return(reds)

# For supports
def classify_sup(points, H, W, RADIUS):
	reds = [0]*7
	points.dropna(inplace=True)
	for point in points:
		if(norm(point - np.array([149,0]))  < RADIUS): # Red base
			reds[6]+=1
		elif(norm(point - np.array([0,149]))    < RADIUS): # Blue base
			reds[3]+=1
		elif(norm(point - np.array([149,149]))  < RADIUS or point[0] > W - W / 10 or point[1] > H - H / 10): # Botlane
			reds[5]+=1
		elif(point[0] < H - point[1] - H/10): # Above midlane
			reds[0]+=1
		elif(point[0] > H - point[1] + H/10): # Below midlane
			if(point[0] < point[1]): # Blue side
				reds[2]+=1
			elif(point[0] > point[1]): # Red side
				reds[1] += 1
		elif(point[0] < H - point[1] + H/10): # Midlane
			reds[4]+=1
	return(reds)

# And for midlaners
def classify_mid(points, H, W, RADIUS):
	reds = [0]*8
	points.dropna(inplace=True)
	for point in points:
		if(norm(point - np.array([149,0]))	< RADIUS): # Red base
			reds[6]+=1
		elif(norm(point - np.array([0,149])) 	< RADIUS): # Blue base
			reds[7]+=1
		elif(norm(point - np.array([149,149])) 	< RADIUS or point[0] > W - W / 10 or point[1] > H - H / 10): # Botlane
			reds[5]+=1
		elif(norm(point - np.array([0,149])) 	< RADIUS or point[0] < W / 10 or point[1] < H / 10): # Toplane
			reds[4]+=1
		elif(point[0] < H - point[1] - H / 10): # Topside jungle
			reds[3]+=1
		elif(point[0] > H - point[1] + H / 10): # Botside jungle
			reds[2]+=1
		elif(point[0] < point[1]): # Past halfway point of midlane
			reds[1]+=1
		elif(point[0] > point[1]): # Behind halfway point of midlane
			reds[0]+=1
	return(reds)