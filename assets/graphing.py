import plotly.express as px
import pandas as pd
import numpy as np
import plotly

from jinja2 import Environment, FileSystemLoader

from PIL import Image
from numpy.linalg import norm as norm
from assets.utils import graph_html
# from assets.classifying import classify_jgl, classify_sup, classify_mid
from assets.constants import TIMESPLITS, TIMESPLITS2, ROLE_DICT

def draw_graphs(df, map_coordinates, video, collect, rows_list, LEAGUE = 'lec'):
    H = map_coordinates[1] - map_coordinates[0]
    W = map_coordinates[3] - map_coordinates[2]
    RADIUS = int(W/2.5) 
    
    graph_dict = {}

    # Graph each level one pattern
    graph_dict['level_ones'] = leveloneplots(df, H, W, LEAGUE, video)
    if(not collect):
        print("Level One graphs complete")

    # Jungle graphs
    graph_dict['blue_jungle'] = jungleplots(df, 'blue')
    graph_dict['red_jungle'] = jungleplots(df, 'red')
    if(not collect):
        print("Jungler Region Maps complete")

    # Support graphs
    graph_dict['blue_support'] = supportplots(df, 'blue')
    graph_dict['red_support'] = supportplots(df, 'red')
    if(not collect):
        print("Support Region Maps complete")

    graph_dict['blue_mid'] = midlaneplots(df, 'blue')
    graph_dict['red_mid'] = midlaneplots(df, 'red')
    if(not collect):
        print("Mid Region Maps complete")

    if(not collect):
        print("Proximity Graphs complete")

    inject_html(graph_dict)
    
    # Midlane graphs
    # midlaneplots(df, H, W, RADIUS, video)
    
    # Graph the proximities
    # rows_list = proximity(df, [0,1,2,3], 4, "blue", "support", video, rows_list) ################
    # rows_list = proximity(df, [0,2,3,4], 1, "blue", "jungle", video, rows_list)
    # rows_list = proximity(df, [5,6,7,8], 9, "red", "support", video, rows_list)
    # rows_list = proximity(df, [5,7,8,9], 6, "red", "jungle", video, rows_list)
    
    # return rows_list

def inject_html(graph_dict):
    file_loader = FileSystemLoader('assets')
    env = Environment(loader=file_loader)
    template = env.get_template('template.html')
    source_html = template.render(**graph_dict)

    with open("filename.html","w") as html_file:
        html_file.write(source_html)

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
    champs = df.champ.unique()
    df = df[df.second <= 90]
    
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        range_x = [0, 1],
        range_y = [1, 0],
        facet_col='role',
        facet_row='side',
        width = 1200,
        height = 500,
        color='second', 
        hover_name = 'second',
        title="<b>Level Ones</b>",
        color_continuous_scale='Oranges',
        labels = {'y':'', 'x':''}
    )

    fig.add_layout_image(
        dict(
            source=Image.open("assets/hq_map.png"),
            xref="x",
            yref="y",
            x=0,
            y=0,
            sizex = 1,
            sizey = 1, 
            sizing="stretch",
            opacity=1,
            layer="below"),
        col='all',
        row='all'
    )

    for i in range(10):
        fig.add_layout_image(
            dict(
                source=Image.open('assets/portraits/%sSquare.png' % champs[i].capitalize().replace("_", "")),
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex=0.15,
                sizey=0.15,
                sizing="stretch",
                opacity=1,
                layer="below"),
            row=round((1+i)/10)+1,
            col=(i % 5)+1
        )

    fig.for_each_annotation(lambda a: a.update(text="<b>{}</b>".format(a.text.split("=")[-1].title())))


    fig.update_xaxes(showgrid=False, showticklabels = False)
    fig.update_yaxes(showgrid=False, showticklabels = False)

    return plotly.offline.plot(fig, output_type = 'div')


def classify_jgl(x, regions):
    point = np.array([x.x, x.y])
    if(norm(point - np.array([0, 0])) < 0.4): # Toplane
        regions[5] += 1
    elif(norm(point - np.array([1,0])) < 0.4): # Red base
        regions[6] += 1
    elif(norm(point - np.array([1,1])) < 0.4): # Botlane
        regions[8]+=1
    elif(norm(point - np.array([0,1])) < 0.4): # Blue base
        regions[7]+=1
    elif(point[0] < 0.9 - point[1]): # Above midlane upper border
        if(point[0] < 1.05*point[1]):
            regions[1]+=1 # Blue side
        else:
            regions[0]+=1 # Red side
    elif(point[0] < 1.1 - point[1]): # Above midlane lower border (in midlane)
        regions[4]+=1 
    elif(point[0] > 1.1 - point[1]): # Below midlane lower border
        if(point[0] < 1.05*point[1]): # Blue side
            regions[2]+=1
        elif(point[0] > 1.05*point[1]): # Red side
            regions[3]+=1

def classify_sup(x, regions):
    point = np.array([x.x, x.y])
    if(norm(point - np.array([1,0])) < 0.4): # Red base
        regions[6] += 1
    elif(norm(point - np.array([0,1])) < 0.4): # Blue base
        regions[3]+=1
    elif(
        norm(point - np.array([1,1])) < 0.4
        or point[0] > 0.9 
        or point[1] > 0.9): # Botlane
            regions[5]+=1
    elif(point[0] < 0.9 - point[1]): # Above midlane upper border
        regions[0]+=1
    elif(point[0] < 1.1 - point[1]): # Above midlane lower border (in midlane)
        regions[4]+=1 
    elif(point[0] > 1.1 - point[1]): # Below midlane lower border
        if(point[0] < 1.05*point[1]): # Blue side
            regions[2]+=1
        elif(point[0] > 1.05*point[1]): # Red side
            regions[1]+=1

# And for midlaners
def classify_mid(points, H, W, RADIUS):
    reds = [0]*8
    points.dropna(inplace=True)
    for point in points:
        if(norm(point - np.array([1,0])) < 0.4): # Red base
            reds[6]+=1
        elif(norm(point - np.array([0,1])) < 0.4): # Blue base
            reds[7]+=1
        elif(norm(point - np.array([1,1])) < 0.4 or point[0] > 0.9 or point[1] > 0.9): # Botlane
            reds[5]+=1
        elif(norm(point - np.array([0,1])) < 0.4 or point[0] < 0.1 or point[1] < 0.1): # Toplane
            reds[4]+=1
        elif(point[0] < H - point[1] - 0.1): # Topside jungle
            reds[3]+=1
        elif(point[0] > H - point[1] + 0.1): # Botside jungle
            reds[2]+=1
        elif(point[0] < point[1]): # Past halfway point of midlane
            reds[1]+=1
        elif(point[0] > point[1]): # Behind halfway point of midlane
            reds[0]+=1
    return(reds)

def jungleplots(df, colour):
    df_jungle = df[(df.role == 'jgl')]
    
    graph_dict = {}
    df_side = df_jungle[df_jungle.side == colour]
    
    subplots = {'x':[0, 480], 'x2':[480, 840], 'x3':[840, 1200]}
    areas = {}
    for subplot in subplots.keys():
        df_timesplit = df_side[(df_side.second >= subplots[subplot][0]) & (df_side.second < subplots[subplot][1])]
        
        regions=[0]*9
        df_timesplit.apply(lambda x : classify_jgl(x, regions), axis=1)
        areas[subplot] = regions

    fig2 = plotly.subplots.make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=(
            '0-8 Minutes',
            '8-14 Minutes',
            '14-20 Minutes'),
    )

    shapes = []
    for subplot in ['x', 'x2', 'x3']:
        reds = areas[subplot]
        reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)), reds))

        # Different colours for each team
        fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"
        shapes.extend([
        dict(
                type="path",
                path = "M 0,0 L 0.5,0.5 L 1,0 Z",
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),
            ),
        dict(
                type="path",
                path = "M 0,0 L 0.5,0.5 L 0,1 Z",
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),
            ),

        dict(
                type="path",
                path = "M 1,1 L 0.5,0.5 L 0,1 Z",
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),
            ),
        dict(
                type="path",
                path = "M 1,1 L 0.5,0.5 L 1,0 Z",
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),
            ),
        dict(
                type="path",
                path = "M 0.1,1 L 1,0.1 L 0.9,0 L 0,0.9 Z",
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
            ),

        dict(
                type="circle",
                yref="y",
                x0=-0.4,
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
                y0=0.4,
                x1=0.4,
                y1=-0.4,
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
            ),
        dict(
                type="circle",
                yref="y",
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[6],reds[6]),
                x0=0.6,
                y0=0.4,
                x1=1.4,
                y1=-0.4,
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
            ),
        dict(
                type="circle",
                yref="y",
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),
                x0=-0.4,
                y0=1.4,
                x1=0.4,
                y1=0.6,
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
            ),
        dict(
                type="circle",
                yref="y",
                fillcolor=('rgba(%s,1)' % fill_team) % (reds[8],reds[8]),
                x0=0.6,
                y0=1.4,
                x1=1.4,
                y1=0.6,
                line={
                    'color':"black",
                    'width':0.5,
                },
                xref=subplot,
            )])

    fig2.update_layout(
        title = "%s Jungler Locations" % colour.title(),
        template = "plotly_white",
        shapes=shapes,
        width=1200,
        height=500,
        coloraxis = dict(colorscale=['white', colour],colorbar=dict(tickmode = 'array', ticktext=['Rarely Here', 'Often Here'], tickvals=[0,1]))
    )

    fig2.update_yaxes(range=[1,0], showgrid=False, showticklabels=False)
    fig2.update_xaxes(range=[0,1], showgrid=False, showticklabels=False)

    fig = px.scatter(x=[0, 100], y=[0, 1], color=[0, 1])

    fig2.add_trace(fig.data[0])
    return plotly.offline.plot(fig2, output_type='div')

    
def supportplots(df, colour):
    df_support = df[(df.role == 'sup')]

    graph_dict = {}
    df_side = df_support[df_support.side == colour]

    subplots = {'x':[0, 480], 'x2':[480, 840], 'x3':[840, 1200]}
    areas = {}
    for subplot in subplots.keys():
        df_timesplit = df_side[(df_side.second >= subplots[subplot][0]) & (df_side.second < subplots[subplot][1])]
        regions=[0]*7
        df_timesplit.apply(lambda x : classify_sup(x, regions), axis=1)
        areas[subplot] = regions

    fig2 = plotly.subplots.make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=(
            '0-8 Minutes',
            '8-14 Minutes',
            '14-20 Minutes'),
    )


    shapes = []
    for subplot in ['x', 'x2', 'x3']:
        reds = areas[subplot]
        reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)), reds))

        # Different colours for each team
        fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"
        shapes.extend([
        dict(
            type="path",
            path = "M 0,0 L 1,1 L 1,0 Z",
            line={
                'color':"black",
                'width':0.5,
            },
            xref=subplot,
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[1],reds[1]),
        ),
        dict(
            type="path",
            path = "M 0,0 L 1,1 L 0,1 Z",
            line={
                'color':"black",
                'width':0.5,
            },
            xref=subplot,
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[2],reds[2]),
        ),
        dict(
            type="path",
            path = "M 0,1 L 1,0 L 0,0 Z",
            line={
                'color':"black",
                'width':0.5,
            },
            xref=subplot,
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[0],reds[0]),
        ),
        dict(
            type="rect",
            x0=0,
            y0=1,
            x1=1,
            y1=0.9,
            line={
                'color':"black",
                'width':0.5,
            },
            xref=subplot,
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
            ),
        dict(
            type="rect",
            x0=0.9,
            y0=1,
            x1=1,
            y1=0,
            line={
                'color':"black",
                'width':0.5,
            },
            xref=subplot,
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
            ),
        dict(
            type="circle",
            yref="y",
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
            x0=0.6,
            y0=1.4,
            x1=1.4,
            y1=0.6,
            xref=subplot,
            line={
                'color':"black",
                'width':0.5,
            }
            ),
        dict( # Plotly doesn't accept the Arc SVG command so this is a workaround to make the lanes look smooth
            type="rect",
            x0=0,
            y0=1,
            x1=1,
            y1=0.905,
            xref=subplot,
            line=dict(
                color=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
                width=2,
            ),
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
            ),
        dict(
            type="rect",
            x0=0.905,
            y0=1,
            x1=1,
            y1=0,
            xref=subplot,
            line=dict(
                color=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
                width=2,
            ),
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[5],reds[5]),
            ),
        dict(
            type="path",
            path = "M 0.1,1 L 1,0.1 L 0.9,0 L 0,0.9 Z",
            line={
                'color':"black",
                'width':0.5,
            },
            xref=subplot,
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[4],reds[4]),
            ),
        dict(
            type="circle",
            xref=subplot,
            yref="y",
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[6],reds[6]),
            x0=0.6,
            y0=0.4,
            x1=1.4,
            y1=-0.4,
            line={
                'color':"black",
                'width':0.5,
            }
        ),
        dict(
            type="circle",
            xref=subplot,
            yref="y",
            fillcolor=('rgba(%s,1)' % fill_team)  % (reds[3],reds[3]),
            x0=-0.4,
            y0=1.4,
            x1=0.4,
            y1=0.6,
            line={
                'color':"black",
                'width':0.5,
            }
            )])

    fig2.update_layout(
        title = "%s Support Locations" % colour.title(),
        template = "plotly_white",
        shapes=shapes,
        width=1200,
        height=500,
        coloraxis = dict(colorscale=['white',colour],colorbar=dict(tickmode = 'array', ticktext=['Rarely Here', 'Often Here'], tickvals=[0,1]))
    )

    fig2.update_yaxes(range=[1,0], showgrid=False, showticklabels=False)
    fig2.update_xaxes(range=[0,1], showgrid=False, showticklabels=False)

    fig = px.scatter(x=[0, 100], y=[0, 1], color=[0, 1])

    fig2.add_trace(fig.data[0])
    return plotly.offline.plot(fig2, output_type='div')


def midlaneplots(df, colour):
    df_jungle = df[(df.role == 'jgl')]
    
    graph_dict = {}
    df_side = df_jungle[df_jungle.side == colour]
    
    subplots = {'x':[0, 480], 'x2':[480, 840], 'x3':[840, 1200]}
    areas = {}
    for subplot in subplots.keys():
        df_timesplit = df_side[(df_side.second >= subplots[subplot][0]) & (df_side.second < subplots[subplot][1])]
        
        regions=[0]*9
        df_timesplit.apply(lambda x : classify_jgl(x, regions), axis=1)
        areas[subplot] = regions

    fig2 = plotly.subplots.make_subplots(
        rows=1, 
        cols=3,
        subplot_titles=(
            '0-8 Minutes',
            '8-14 Minutes',
            '14-20 Minutes'),
    )

    shapes = []
    for subplot in ['x', 'x2', 'x3']:
        reds = areas[subplot]
        reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)), reds))

        # Different colours for each team
        fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"
        shapes.extend([
                dict(
                        type="path",
                        path = "M 0,0 L 1,1 L 1,0 Z",
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[0],reds[0]),
                    ),
                dict(
                        type="path",
                        path = "M 0,0 L 1,1 L 0,1 Z",
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[1],reds[1]),
                    ),
                dict(
                        type="path",
                        path = "M 0.1,1 L 1,0.1 L 1,1 Z",
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[2],reds[2]),
                    ),
                    
                dict(
                        type="path",
                        path = "M 0,0.9 L 0.9,0 L 0,0 Z",
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[3],reds[3]),
                    ),
                dict(
                        type="rect",
                        x0=0,
                        y0=1,
                        x1=0.1,
                        y1=0,
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
                    ),
                dict(
                        type="rect",
                        x0=0,
                        y0=0.1,
                        x1=1,
                        y1=0,
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
                    ),
                dict(
                        type="circle",
                        yref="y",
                        x0=-0.4,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
                        y0=0.4,
                        x1=0.4,
                        y1=-0.4,
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                    ),
                dict(
                        type="rect",
                        x0=0,
                        y0=0.095,
                        x1=1,
                        y1=0,
                        line=dict(
                            color=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
                            width=2,
                        ),
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
                    ),
                dict(
                        type="rect",
                        x0=0,
                        y0=1,
                        x1=0.095,
                        y1=0,
                        line=dict(
                            color=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
                            width=2,
                        ),
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[4],reds[4]),
                    ),
                dict(
                        type="rect",
                        x0=0,
                        y0=1,
                        x1=1,
                        y1=0.9,
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
                    ),
                dict(
                        type="rect",
                        x0=0.9,
                        y0=1,
                        x1=1,
                        y1=0,
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
                    ),
                dict(
                        type="circle",
                        xref=subplot,
                        yref="y",
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
                        x0=0.6,
                        y0=1.4,
                        x1=1.4,
                        y1=0.6,
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                    ),
                dict(
                        type="rect",
                        x0=0,
                        y0=1,
                        x1=1,
                        y1=0.905,
                        line=dict(
                            color=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
                            width=2,
                        ),
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
                    ),
                dict(
                        type="rect",
                        x0=0.905,
                        y0=1,
                        x1=1,
                        y1=0,
                        line=dict(
                            color=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
                            width=2,
                        ),
                        xref=subplot,
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[5],reds[5]),
                    ),
                dict(
                        type="circle",
                        xref=subplot,
                        yref="y",
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[6],reds[6]),
                        x0=0.6,
                        y0=0.4,
                        x1=1.4,
                        y1=-0.4,
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                    ),
                dict(
                        type="circle",
                        xref=subplot,
                        yref="y",
                        fillcolor=('rgba(%s,1)' % fill_team) % (reds[7],reds[7]),
                        x0=-0.4,
                        y0=1.4,
                        x1=0.4,
                        y1=0.6,
                        line={
                            'color':"black",
                            'width':0.5,
                        },
                    )
                    
                ])

    fig2.update_layout(
        title = "%s Midlane Locations" % colour.title(),
        template = "plotly_white",
        shapes=shapes,
        width=1200,
        height=500,
        coloraxis = dict(colorscale=['white', colour],colorbar=dict(tickmode = 'array', ticktext=['Rarely Here', 'Often Here'], tickvals=[0,1]))
    )

    fig2.update_yaxes(range=[1,0], showgrid=False, showticklabels=False)
    fig2.update_xaxes(range=[0,1], showgrid=False, showticklabels=False)

    fig = px.scatter(x=[0, 100], y=[0, 1], color=[0, 1])

    fig2.add_trace(fig.data[0])
    return plotly.offline.plot(fig2, output_type='div')
    # colour = "blue"
    
    # for col in [df.columns[n] for n in [2,7]]:
    #     fill_team = "255, %d, %d" if colour == "red" else "%d, %d, 255"
    #     for times in TIMESPLITS.keys():
    #         reds = classify_mid(df[col][df['Seconds'] <= times], H, W, RADIUS)
    #         reds = list(map(lambda x : 255-255*(x - min(reds))/(max(reds)-min(reds)), reds))
            
    #         fig = px.scatter(
    #                 x = [0], 
    #                 y = [0],
    #                 range_x = [0,W],
    #                 range_y = [H, 0],
    #                 width = 800,
    #                 height = 800)


    #         fig.update_layout(
    #                 template = "plotly_white",
    #                 xaxis_showgrid = False,
    #                 yaxis_showgrid = False
    #                 )

    #         fig.update_xaxes(showticklabels = False, title_text = "")
    #         fig.update_yaxes(showticklabels = False, title_text = "")

    #         fig.update_layout(
                # shapes=)
    #         fig.update_layout(
    #             title = "%s: %s" % (col.capitalize(), TIMESPLITS[times]),
    #             template = "plotly_white",
    #             xaxis_showgrid = False,
    #             yaxis_showgrid = False
    #             )
    #         graph_html(plotly.offline.plot(fig, output_type = 'div'), video, colour, "mid/%s_%s" % (col,TIMESPLITS[times]))
    #     colour = "red"
    