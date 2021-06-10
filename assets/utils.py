import numpy as np

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