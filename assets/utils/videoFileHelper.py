import os

VALID_VIDEO_FORMATS = ["WEBM", "MPG", "MP2", "MPEG", "MPE", "MPV", "OGG", "MP4", "M4P", "M4V", "AVI", "WMV", "MOV", "QT", "FLV", "SWF", "AVCHD"]

def clean_for_directory(video_title : str) -> str:
    for ch in ['*', '.', '"', '/', '\\', ':', ';', '|', ',']:
        if ch in video_title:
            video_title = video_title.replace(ch, '')
        video_title = video_title.replace(' ', '_')

    return video_title

def parse_local_files(url : str) -> dict or None:
    videos = {}
    if os.path.isdir(url):
        for file in os.listdir(url):
            if file.split('.')[-1] in VALID_VIDEO_FORMATS:
                # get the file's name
                video_name = clean_for_directory(file)
                videos[video_name] = os.path.join(url, file)
            else:
                print(f'Invalid file format: {file}')
    elif os.path.isfile(url):
        if url.split('.')[-1] in VALID_VIDEO_FORMATS:
            video_name = clean_for_directory(url)
            videos[video_name] = url
            return videos
        else:
            print(f'Invalid file format: {url}')
            return None