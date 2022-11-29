"""Tools to help extract data."""
import os

import numpy as np

from moviepy.editor import *
from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi


def download_video(url: str, save_to: str, ext: str = 'mp4', res: str = '720p'):
    """Download YouTube video provided its link.
    
    Parameters
    ----------
    url : str
        Link to YouTube video.
    save_to : str
        Path to folder where to save video.
        You don't need to specify file name and its extension, just folder.
    ext : str, default='mp4'
        Video file extension. Refer to pytube docs for more info.
    res : str, default='720p'
        Video file resolution. Refer to pytube docs for more info.
        
    Returns
    -------
    title : str
        Title of YouTube video.
    """
    yt = YouTube(url)
    mp4_files = yt.streams.filter(file_extension=ext)
    mp4_720p_files = mp4_files.get_by_resolution(res)
    mp4_720p_files.download(save_to)
    
    title = yt.streams[0].title
    
    return title
    
    
def url_to_id(url: str):
    """Convert YouTube url to id (only full links are valid)."""
    video_start = url.find('.be/')
    video_end = url.find('?')
    video_id = url[video_start+4:video_end]
    
    return video_id


def extract_captions(video_id: str, lang: str = 'ru'):
    """Extract YouTube video captions.
    
    Parameters
    ----------
    video_id : str
        YouTube video id.
    lang : str, default='ru'
        Language of captions.
        
    Returns
    -------
    captions : list of dict
        Captions with timestamps.
    """
    captions = YouTubeTranscriptApi.get_transcript(video_id, languages=[lang])
    
    return captions


def video_to_chopped_audio(video: str, path: str, sr=16000, chop_by=30, ext='ogg'):
    """Convert video file to audio.
    
    Parameters
    ----------
    video : str
        Path to existing video file (including its name and extension).
    path : str
        Path to audio folder (do not include extension).
    sr : int
        Sample rate for audio file.
    chop_by : int
        How many seconds to include in one chopped audio file.
    ext : str
        Audio file extension. Refer to moviepy docs for available extensions.
    """
    video_file = VideoFileClip(video)
    video_duration = video_file.duration
    
    if not os.path.isdir(path):
        os.mkdir(path)
    
    start = np.arange(0, video_duration + chop_by - 1, chop_by)[:-1]
    stop = np.arange(0, video_duration + chop_by - 1, chop_by)[1:]
    stop[-1] = video_duration
    
    counter = 0
    for begin, end in zip(start, stop):
        temp_file = video_file.subclip(t_start=begin, t_end=end)
        temp_file.audio.write_audiofile(path + f'chopped_{counter}.{ext}', fps=sr)
        counter += 1
