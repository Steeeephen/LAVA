import pandas as pd

from assets.graphing import GraphsOperator

g = GraphsOperator()

df = pd.read_csv('output/positions/Liquid_vs_G2_Esports_Game_1_-_MSI_2019_Finals_-_TL_vs_G2_G1.csv')

g.draw_graphs(df=df, video='okok')