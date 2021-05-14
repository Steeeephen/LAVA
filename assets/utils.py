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