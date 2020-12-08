import plotly.express as px
import pandas as pd
import numpy as np
import plotly

from PIL import Image
from numpy.linalg import norm as norm
from assets.utils import graph_html
from assets.classifying import classify_jgl, classify_sup, classify_mid
from assets.constants import TIMESPLITS, TIMESPLITS2, ROLE_DICT

def draw_graphs(df, H, W, LEAGUE, video, collect, rows_list):
	RADIUS = int(W/2.5)	
	# Graph each level one pattern
	leveloneplots(df, H, W, LEAGUE, video)
	if(not collect):
		print("Level One graphs complete")

	# Jungle graphs
	jungleplots(df, H, W, RADIUS, video)
	if(not collect):
		print("Jungler Region Maps complete")

	# Support graphs
	supportplots(df, H, W, RADIUS, video)
	if(not collect):
		print("Support Region Maps complete")

	# Midlane graphs
	midlaneplots(df, H, W, RADIUS, video)
	if(not collect):
		print("Mid Region Maps complete")

	# Graph the proximities
	rows_list = proximity(df, [0,1,2,3], 4, "blue", "support", video, rows_list) ################
	rows_list = proximity(df, [0,2,3,4], 1, "blue", "jungle", video, rows_list)
	rows_list = proximity(df, [5,6,7,8], 9, "red", "support", video, rows_list)
	rows_list = proximity(df, [5,7,8,9], 6, "red", "jungle", video, rows_list)
	if(not collect):
		print("Proximity Graphs complete")
	return rows_list


# This will graph the proximities for a given role and side, showing how close two players were throughout the game
def proximity(df, l, t, side, role, video, rows_list):
	roles = list(ROLE_DICT.keys())
	plots = ""
	role1 = df.columns[t]
	for ally in l: # For each allied champion
		count = 0
		champ = df.columns[ally]
		champ_to_check = pd.DataFrame((df[df.columns[t]]).apply(lambda x : np.array(x)))
		champ_teammate =pd.DataFrame((df[champ]).apply(lambda x : np.array(x)))

		diffs = [0]*len(champ_teammate)
		for j in range(len(champ_teammate)): # For every pair of points gathered
			dist = norm(champ_to_check[df.columns[t]].tolist()[j] - champ_teammate[champ].tolist()[j]) # Get the distance between the two
			diffs[j] = dist

			# If within 50 units, they're considered 'close'
			if(dist < 50):
				count+=1
		
		# Graph the distances over time
		fig2 = px.line(y=diffs, title = "%.2f%% Proximity" % (100*count/len(df['Seconds'])), x=df["Seconds"]/60)
		fig2.update_yaxes(title="Distance to: %s" % champ.capitalize())
		fig2.add_shape( 
				type="line",
				x0=df['Seconds'].min()/60,
				y0=50,
				x1=20,
				y1=50,
				line=dict(
					color="MediumPurple",
					width=4,
					dash="dot",
				)
		)
		fig2.update_xaxes(rangeslider_visible=True, title = "Minute")
		plots += plotly.offline.plot(fig2, output_type = 'div')
	
		rows_list.append({"video": video, "side":side, "role":role, "target":roles[ally%5], "player":role1, "teammate":df.columns[ally], "proximity":100*count/len(df['Seconds'])})
	
	graph_html(plots, video, side, role + "_proximities")
	return rows_list
	
	
def leveloneplots(df, H, W, LEAGUE, video):
	colour = "blue"
	cols = df.columns[:-1]

	for col_i, col in enumerate(cols):
		if col_i > 4: # If we graphed all 5 blue side champions
			colour = "red"

		temp_df = df[df['Seconds'] <= 90]
		champ = temp_df[col]
		axes = list(zip(*champ))

		fig = px.scatter(
			temp_df,
			x = axes[0], 
			y = axes[1],
			range_x = [0, W],
			range_y = [H, 0],
			width = 800,
			height = 800,
			hover_name = 'Seconds',
			color = 'Seconds',
			color_continuous_scale = "Oranges")

		fig.add_layout_image(
			dict(
				source=Image.open("assets/%s/%s.png" % (LEAGUE, LEAGUE)),
				xref="x",
				yref="y",
				x=0,
				y=0,
				sizex = W,
				sizey = H, 
				sizing="stretch",
				opacity=1,
				layer="below")
		)

		fig.add_layout_image(
				dict(
					source=Image.open('assets/portraits/%sSquare.png' % col.capitalize().replace("_", "")),
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
		graph_html(plotly.offline.plot(fig, output_type = 'div'), video, colour, "levelone/%s" % col)


def jungleplots(df, H, W, RADIUS, video):
	colour = "blue"

	for col in [df.columns[n] for n in [1,6]]:
		# We split these up into our important intervals
		for times in TIMESPLITS.keys():

			# Get time intervals
			times_floor = TIMESPLITS2[times]

			# Classify points
			reds = classify_jgl(df[col][(df['Seconds'] <= times) & (df['Seconds'] >= times_floor)], H, W, RADIUS)
			
			# We scale our points so that they fall nicely on [0,255], meaning easier to read graphs
			reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)), reds))
			fig = px.scatter(
					x = [0], 
					y = [0],
					range_x = [0, W],
					range_y = [H, 0],
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
						path = "M 0,0 L %d,%d L %d,0 Z" % (W/2,H/2,W),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),
					),
				dict(
						type="path",
						path = "M 0,0 L %d,%d L 0,%d Z" % (W/2,H/2,W),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),
					),
				
				dict(
						type="path",
						path = "M %d,%d L %d,%d L 0,%d Z" % (W,H,W/2, H/2,H),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),
					),
				dict(
						type="path",
						path = "M %d,%d L %d,%d L %d,0 Z" % (W,H,W/2,H/2,W),
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),
					),
				dict(
						type="path",
						path = "M %d,%d L %d,%d L %d,0 L 0,%d Z" % (W/10,H, W,H/10,W-W/10,H-H/10),
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
						x0=-RADIUS,
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
						y0=RADIUS,
						x1=RADIUS,
						y1=-RADIUS,
						line_color="white",
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[6],reds[6]),
						x0=W-RADIUS,
						y0=RADIUS,
						x1=W+RADIUS,
						y1=-RADIUS,
						line_color="white",
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),
						x0=-RADIUS,
						y0=H+RADIUS,
						x1=RADIUS,
						y1=H-RADIUS,
						line_color="white",
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[8],reds[8]),
						x0=W-RADIUS,
						y0=H+RADIUS,
						x1=W+RADIUS,
						y1=H-RADIUS,
						line_color="white",
					)])
			fig.update_layout(
				title = "%s: %s" % (col.capitalize(), TIMESPLITS[times]),
				template = "plotly_white",
				xaxis_showgrid = False,
				yaxis_showgrid = False
				)
			graph_html(plotly.offline.plot(fig, output_type = 'div'), video, colour, "jungle/%s_%s" % (col, TIMESPLITS[times]))
		colour = "red"
	
def supportplots(df, H, W, RADIUS, video):
	colour = "blue"
	
	for col in [df.columns[n] for n in [4,9]]:
		for times in TIMESPLITS.keys():
			reds = classify_sup(df[col][df['Seconds'] <= times], H, W, RADIUS)
			reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)-min(reds)), reds))
			
			fig = px.scatter(
					x = [0], 
					y = [0],
					range_x = [0, W],
					range_y = [H, 0],
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
							path = "M 0,0 L %d,%d L %d,0 Z" % (W, H, W),
							line=dict(
								color='white',
								width=2,
							),
							fillcolor=('rgba(%s,1)' % fill_team)  % (reds[1],reds[1]),
					),
					dict(
							type="path",
							path = "M 0,0 L %d,%d L 0,%d Z" % (W, H, H),
							line=dict(
								color='white',
								width=2,
							),
							fillcolor=('rgba(%s,1)' % fill_team)  % (reds[2],reds[2]),
					),
					dict(
							type="path",
							path = "M 0,%d L %d,0 L 0,0 Z" % (H, W),
							line=dict(
								color='white',
								width=2,
							),
							fillcolor=('rgba(%s,1)' % fill_team)  % (reds[0],reds[0]),
					),
					dict(
							type="rect",
							x0=0,
							y0=H,
							x1=W,
							y1=H-H/10,
							line=dict(
								color="white",
								width=2,
							),
							fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
						),
					dict(
							type="rect",
							x0=W-W/10,
							y0=H,
							x1=W,
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
							x0=W-RADIUS,
							y0=H+RADIUS,
							x1=W+RADIUS,
							y1=H-RADIUS,
							line_color='white',
						),
					dict( # Plotly doesn't accept the Arc SVG command so this is a workaround to make the lanes look smooth
							type="rect",
							x0=0,
							y0=H,
							x1=W,
							y1=H-H/10+0.5,
							line=dict(
								color=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
								width=2,
							),
							fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
						),
					dict(
							type="rect",
							x0=W-W/10+0.5,
							y0=H,
							x1=W,
							y1=0,
							line=dict(
								color=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
								width=2,
							),
							fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
						),
					dict(
							type="path",
							path = "M %d,%d L %d,%d L %d,0 L 0,%d Z" % (W/10,H, W,H/10,W-W/10,H-H/10),
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
							x0=W-RADIUS,
							y0=RADIUS,
							x1=W+RADIUS,
							y1=-RADIUS,
							line_color='white',
						),
					dict(
							type="circle",
							xref="x",
							yref="y",
							fillcolor=('rgba(%s,1)' % fill_team)  % (reds[3],reds[3]),
							x0=-RADIUS,
							y0=H+RADIUS,
							x1=RADIUS,
							y1=H-RADIUS,
							line_color="white",
						)]) 
			fig.update_layout(
				title = "%s: %s" % (col.capitalize(), TIMESPLITS[times]),
				template = "plotly_white",
				xaxis_showgrid = False,
				yaxis_showgrid = False
				)
			graph_html(plotly.offline.plot(fig, output_type = 'div'), video, colour, "support/%s_%s" % (col,TIMESPLITS[times]))
		colour = "red"
	
def midlaneplots(df, H, W, RADIUS, video):
	colour = "blue"
	
	for col in [df.columns[n] for n in [2,7]]:
		fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"
		for times in TIMESPLITS.keys():
			reds = classify_mid(df[col][df['Seconds'] <= times], H, W, RADIUS)
			reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)-min(reds)), reds))
			
			fig = px.scatter(
					x = [0], 
					y = [0],
					range_x = [0,W],
					range_y = [H, 0],
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
						path = "M 0,0 L %d,%d L %d,0 Z" % (W,H,W),
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),
					),
				dict(
						type="path",
						path = "M 0,0 L %d,%d L 0,%d Z" % (W,H,H),
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),
					),
				dict(
						type="path",
						path = "M %d,%d L %d,%d L %d,%d Z" % (W/10,H,W,H/10,W,H),
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),
					),
					
				dict(
						type="path",
						path = "M 0,%d L %d,%d L 0,0 Z" % (H-H/10,W-W/10,0),
						line=dict(
							color='white',
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),
					),
				dict(
						type="rect",
						x0=0,
						y0=H,
						x1=W/10,
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
						y0=H/10,
						x1=W,
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
						x0=-RADIUS,
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
						y0=RADIUS,
						x1=RADIUS,
						y1=-RADIUS,
						line_color='white',
					),
				dict(
						type="rect",
						x0=0,
						y0=H/10-0.5,
						x1=W,
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
						y0=H,
						x1=W/10-0.5,
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
						y0=H,
						x1=W,
						y1=H-H/10,
						line=dict(
							color="white",
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
					),
				dict(
						type="rect",
						x0=W-W/10,
						y0=H,
						x1=W,
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
						x0=W-RADIUS,
						y0=H+RADIUS,
						x1=W+RADIUS,
						y1=H-RADIUS,
						line_color='white',
					),
				dict(
						type="rect",
						x0=0,
						y0=H,
						x1=W,
						y1=H-H/10+0.5,
						line=dict(
							color=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
							width=2,
						),
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
					),
				dict(
						type="rect",
						x0=W-W/10+0.5,
						y0=H,
						x1=W,
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
						x0=W-RADIUS,
						y0=RADIUS,
						x1=W+RADIUS,
						y1=-RADIUS,
						line_color='white',
					),
				dict(
						type="circle",
						xref="x",
						yref="y",
						fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),
						x0=-RADIUS,
						y0=H+RADIUS,
						x1=RADIUS,
						y1=H-RADIUS,
						line_color="white",
					)
					
				])
			fig.update_layout(
				title = "%s: %s" % (col.capitalize(), TIMESPLITS[times]),
				template = "plotly_white",
				xaxis_showgrid = False,
				yaxis_showgrid = False
				)
			graph_html(plotly.offline.plot(fig, output_type = 'div'), video, colour, "mid/%s_%s" % (col,TIMESPLITS[times]))
		colour = "red"
