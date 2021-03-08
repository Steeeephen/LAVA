from assets.data_gathering import LolTracker

import pandas as pd

a = LolTracker()

# df = a.gather_data('test.mp4', local=True)
# df = a.gather_data('https://www.youtube.com/playlist?list=PLTCk8PVh_Zwk1rYwLf7IVU5-itu9QJdMb', playlist=True)

# print(df)

# df.to_csv('ok.csv')

df = pd.read_csv("ok.csv")

print(a.proximity_values(df))