from yt_dlp import YoutubeDL

from .video_file_helper import clean_for_directory


def is_valid_youtube_url(url) -> bool:
    return 'youtube' in url and '/watch?v=' in url

def parse_youtube_url(url : str) -> dict:
    """
    """
    videos = {}
    ydl_opts = {
        'format': '22',
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(
            url,
            download=False
        )
        
        video_info = ydl.sanitize_info(info)

        if video_info['_type'] == 'playlist':
            for video in video_info['entries']:
                video_name = clean_for_directory(video['title'])

                videos[video_name] = video_info['formats'][-1]['url']
        else:
            video_name = clean_for_directory(video_info['title'])

            videos[video_name] = video_info['url']

    return videos
