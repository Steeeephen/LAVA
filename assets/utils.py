import cv2
import numpy as np
import pandas as pd
import os
from sympy.geometry import Point, Circle, intersection, Line
from sympy import N

from assets.constants import *

def clean_for_directory(video_title):
    for ch in ['*', '.', '"', '/', '\\', ':', ';', '|', ',']:
        if ch in video_title:
            video_title = video_title.replace(ch, '')
        video_title = video_title.replace(' ', '_')

    return video_title

def get_header_borders(frame_shape):
    frame_height, frame_width, _ = frame_shape
    header_height = frame_height//15
    header_width_left = 6*(frame_width//13)
    header_width_right = 7*(frame_width//13)

    header_borders = (
        slice(0, header_height), 
        slice(header_width_left, header_width_right)
    )

    return header_borders

# def data_collect():
#     directory = os.listdir("output")
#     videos = []
#     for video in directory:
#         if os.path.exists("output/%s/positions.csv" % video):
#             videos.append(video)

#     roles = ['top', 'jgl', 'mid', 'adc', 'sup']*2

#     # Read first dataframe
#     df = pd.read_csv("output/%s/positions.csv" % videos[0])

#     x = df['Seconds']

#     # Use seconds as index and drop column
#     df.index = df['Seconds']
#     df.drop('Seconds', axis=1, inplace=True)
#     df = df.T  # Transpose matrix

#     # Remove duplicated seconds
#     df = df.loc[:, ~df.columns.duplicated()]

#     # Fill in missing seconds with NaN
#     buffer = [i for i in range(1200) if i not in x.tolist()]
#     for i in buffer:
#         df[str(i)] = [(np.nan, np.nan)]*10

#     # Sort columns by seconds
#     df = df[df.columns[np.argsort(df.columns.astype(int))]]

#     # Add context columns
#     df['video'] = videos[0]
#     df['champ'] = df.index
#     df['roles'] = (roles)

#     # Reorder columns
#     y = df.columns[-3:].tolist()
#     y.extend(df.columns[:-3].tolist())
#     df = df[y]
#     df.index = range(10)

#     df.columns = df.columns.astype(str)

#     # Repeat process for every game recorded and add to database
#     for video in videos[1:]:
#         df1 = pd.read_csv("output/%s/positions.csv" % video)
#         x = df1['Seconds']

#         df1.index = df1['Seconds']
#         df1.drop('Seconds', axis=1, inplace=True)
#         df1 = df1.T

#         df1 = df1.loc[:, ~df1.columns.duplicated()]

#         buffer = [i for i in range(1200) if i not in x.tolist()]
#         for i in buffer:
#             df1[str(i)] = [(np.nan, np.nan)]*10

#         df1 = df1[df1.columns[np.argsort(df1.columns.astype(int))]]

#         df1['video'] = video
#         df1['champ'] = df1.index
#         df1['roles'] = (roles)

#         df1.columns = df1.columns.astype(str)

#         y = df1.columns[-3:].tolist()
#         y.extend(df1.columns[:-3].tolist())
#         df1 = df1[y]
#         df1.index = range(10)
#         df = pd.concat([df, df1], ignore_index=True)

#     # Replace '_' character from champ names
#     # Caused by champ having multiple splash arts after a visual update
#     df['champ'] = df['champ'].apply(lambda x: x.replace("_", ""))
#     return(df)
