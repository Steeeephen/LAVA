import numpy as np
import os
import cv2

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

def decodeText(scores):
    text = ""
    alphabet = "0123456789abcdefghijklmn0pqr5tuvwxyz"
    for i in range(scores.shape[0]):
        c = np.argmax(scores[i][0])
        if c != 0:
            text += alphabet[c - 1]
        else:
            text += '-'

    # adjacent same letters as well as background text must be removed to get the final output
    char_list = []
    for i in range(len(text)):
        if text[i] != '-' and (not (i > 0 and text[i] == text[i - 1])):
            char_list.append(text[i])
    return ''.join(char_list)

def get_digit_templates(overlay):
    digits_path = os.path.join(
        'assets',
        'tracking',
        'digits',
        overlay
    )

    digit_templates = {}

    for digit in range(10):
        digit_image_path = os.path.join(
            digits_path,
            f'{digit}.png'
        )

        digit_templates[str(digit)] = cv2.imread(digit_image_path, 0)

    return digit_templates

# Function to recursively clean up and convert the timer reading to seconds.
def timer(time_read, last=""):
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