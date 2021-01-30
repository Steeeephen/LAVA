import plotly.express as px
import pandas as pd
import numpy as np
import plotly
from scipy.spatial.distance import pdist

from jinja2 import Environment, FileSystemLoader

from PIL import Image
from numpy.linalg import norm as norm
from assets.utils import graph_html

def draw_graphs(df, video, collect):
    graph_dict = {}

    # Graph each level one pattern
    graph_dict['level_ones'] = leveloneplots(df)#, H, W, LEAGUE, video)
    if(not collect):
        print("Level One graphs complete")

    # Role Region Maps
    for role in ['jungle', 'support', 'mid']:
        for side in ['blue', 'red']:
            graph_dict["%s_%s" % (side, role)] = eval("%splots(df, '%s')" % (role, side))
        if(not collect):
            print("%s Region Maps Complete" % role.title())

    graph_dict['prox_blue'], graph_dict['prox_red'] = proximities(df)
    
    inject_html(graph_dict)

def inject_html(graph_dict):
    file_loader = FileSystemLoader('assets')
    env = Environment(loader=file_loader)
    template = env.get_template('template.html')
    source_html = template.render(**graph_dict)

    with open("filename.html","w") as html_file:
        html_file.write(source_html)

def for_each_side(df_side):
    combos = np.array([
        'adc_jgl',
        'adc_mid',
        'adc_sup',
        'adc_top',
        'jgl_mid',
        'jgl_sup',
        'jgl_top',
        'mid_sup',
        'mid_top',
        'sup_top'
    ])
    
    proximities = dict([(combo,0) for combo in combos])
    seconds = df_side.second.unique()
    for second in seconds:
        df_second = df_side[df_side.second == second]
        df_array = df_second[["x", "y"]].to_numpy()
        dist_mat = pdist(df_array)
        # pd.DataFrame(dist_mat, index=l)
        close_proximities = combos[np.where(dist_mat < 0.15)]
        for combo in close_proximities:
            proximities[combo] += 1
    return pd.Series(proximities)

def proximities(df):
    df.drop_duplicates(subset=['second', 'side', 'role'], inplace=True)
    df.sort_values(['second', 'side', 'role'], inplace=True)
    prox = df.groupby('side').apply(for_each_side).T
    prox.blue = (prox.blue - prox.blue.min()) / (prox.blue.max() - prox.blue.min())
    prox.red = (prox.red - prox.red.min()) / (prox.red.max() - prox.red.min())
    prox.loc[prox.blue < 0.35, 'blue'] = 0
    prox.loc[prox.red < 0.35, 'red'] = 0
    
    subplots = {'x':[0, 480], 'x2':[480, 840], 'x3':[840, 1200]}
    graphs = {}
    for side in ['red', 'blue']:
        fig = plotly.subplots.make_subplots(rows=1, cols=3, subplot_titles=[
            '0-8 Minutes',
            '8-14 Minutes',
            '14-20 Minutes'])
        for i, subplot in enumerate(subplots.keys()):
            df_means = df[(df.side==side) & (df.second > subplots[subplot][0]) & (df.second < subplots[subplot][1])].groupby(['champ', 'role'])['x','y'].mean()

            fig2 = px.scatter(
                df_means,
                x='x', 
                y='y',
                range_x = [0, 1],
                range_y = [1, 0],
                width = 800,
                color_discrete_sequence=['white'],
                height = 800
            )
            champs = df_means.reset_index().champ.tolist()
            combos = prox.index.tolist()
            df_means_reset = df_means.reset_index()

            for champ in champs:
                fig.add_layout_image(
                    dict(
                        source=Image.open("../loltracker/assets/portraits/%sSquare.png" % champ.title()),
                        xref=subplot,
                        yref="y",
                        x=df_means['x'][champ][0]-0.05,
                        y=df_means['y'][champ][0]-0.05,
                        sizex = 0.1,
                        sizey = 0.1, 
                        sizing="stretch",
                        opacity=1,
                        layer="above")
                )

            for combo in combos:
                combo_roles = combo.split('_')
                dft=(df_means_reset[df_means_reset.role.isin(combo_roles)][['x','y']]).values.tolist()

                fig.add_shape(type="line", layer='above', xref=subplot,
                    x0=dft[0][0], y0=dft[0][1], x1=dft[1][0], y1=dft[1][1],
                    line=dict(color="white",width=20*prox.loc[combo][side])
                )

            fig.add_trace(fig2.data[0], col=i+1, row=1)
        fig.add_layout_image(
            dict(
                source=Image.open("../loltracker/assets/map2.png"),
                xref="x",
                yref="y",
                x=0,
                y=0,
                sizex = 1,
                sizey = 1, 
                sizing="stretch",
                opacity=1,
                layer="below"),
            col="all",
            row=1
        )

        fig.update_layout(
            height=500,
            width=1000,
            title_text="Proximities and Average Positions"
        )

        fig.update_yaxes(range=[1,0], showgrid=False, showticklabels=False)
        fig.update_xaxes(range=[0,1], showgrid=False, showticklabels=False)
        graphs[side] = plotly.offline.plot(fig, output_type = 'div')
    return graphs['blue'], graphs['red']
    
def leveloneplots(df):
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
            source=Image.open("assets/map2.png"),
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


def midplots(df, colour):
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