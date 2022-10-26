import cv2

from utils import overlay_utils

# for i in overlay_utils.get_assets('summoners'):
#     print(i)

# print(overlay_utils.get_summoner_spells())

# print(overlay_utils.get_champion_portraits('blue').keys())
# print(overlay_utils.get_champion_portraits('blue')['nilah'])
# print(overlay_utils.get_champion_portraits('green')['nilah'])
# print(overlay_utils.get_champion_portraits('red'))

x = overlay_utils.get_champion_minimap_icon('nilah')

cv2.imshow('test', x)
cv2.waitKey()

x = overlay_utils.get_champion_minimap_icon('testefds')
