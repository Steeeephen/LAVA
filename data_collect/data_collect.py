"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------

Collect data

This script is for collecting all the location data gathered by the program and stack it in a single csv
It scans through each positions.csv file and outputs a collected_data.csv file with all the data in one place

--------------------------------------------------------------------------------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import os

videos = os.listdir("../output")

# In case data_collection.py was run
if(os.path.exists("../output/proximities.csv")):
	videos.remove("proximities.csv")

roles = ['top','jgl','mid','adc','sup']*2

# Read first dataframe
df = pd.read_csv("../output/%s/positions.csv" % videos[0])

x = df['Seconds']

# Use seconds as index and drop column
df.index = df['Seconds']
df.drop('Seconds', axis=1, inplace=True)
df = df.T # Transpose matrix

# Remove duplicated seconds
df = df.loc[:,~df.columns.duplicated()]

# Fill in missing seconds with NaN
buffer = [i for i in range(1200) if i not in x.tolist()]
for i in buffer:
    df[str(i)] = [(np.nan,np.nan)]*10

# Sort columns by seconds
df = df[df.columns[np.argsort(df.columns.astype(int))]]
    
# Add context columns
df['video'] = videos[0]
df['champ'] = df.index
df['roles'] = (roles)
    
# Reorder columns
y = df.columns[-3:].tolist()
y.extend(df.columns[:-3].tolist())
df = df[y]
df.index = range(10)

df.columns = df.columns.astype(str)

# Repeat process for every game recorded and add to database
for video in videos[1:]:
    df1 = pd.read_csv("../output/%s/positions.csv" % video)
    x = df1['Seconds']

    df1.index = df1['Seconds']
    df1.drop('Seconds', axis=1, inplace=True)
    df1 = df1.T

    df1 = df1.loc[:,~df1.columns.duplicated()]


    buffer = [i for i in range(1200) if i not in x.tolist()]
    for i in buffer:
        df1[str(i)] = [(np.nan,np.nan)]*10

    df1 = df1[df1.columns[np.argsort(df1.columns.astype(int))]]

    df1['video'] = video
    df1['champ'] = df1.index
    df1['roles'] = (roles)

    df1.columns = df1.columns.astype(str)


    y = df1.columns[-3:].tolist()
    y.extend(df1.columns[:-3].tolist())
    df1 = df1[y]
    df1.index = range(10)
    df = pd.concat([df,df1], ignore_index=True)

# Should be (x,1203) with x = 10*(number of games)
print(df.shape)

# Save data
df.to_csv("../output/collected_data.csv")