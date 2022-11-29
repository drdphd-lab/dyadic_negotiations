from collections import defaultdict

from fer import Video, FER

from joblib import Parallel, delayed


def perform_fer(path, mtcnn=True):
    """Recognize emotions from a given video.
    
    Parameters
    ----------
    path : str
        Path to existing video file (including its name and extension).
    mtcnn : bool
        Whether to use MTCNN (more accurate but slower). Used to detect face.
        
    Returns
    -------
    emotions : pandas DataFrame
        Table with bounding box coordinates and probabilities for each emotion.
    """
    video = Video(path)
    
    detector = FER(mtcnn=True)
    emotions = video.analyze(
        detector=detector,
        save_frames=False,
        save_video=False,
        zip_images=False,
    )
    
    return emotions


def perform_fer_parallel(path_lst: list, mtcnn=True, **parallel_kwargs):
    """Extract emotions in parallel manner.
    
    Parameters
    ----------
    path_lst : list
        Paths to video files.
    mtcnn : bool
        Whether to use MTCNN (more accurate but slower). Used to detect face.     
    **parallel_kwargs : dict, optional
        Keyword arguments for joblib.Parallel class.
    
    Returns
    -------
    result : list of dict
        Words, theirs timestamps and probabilities.
    """
    with Parallel(**parallel_kwargs) as parallel:
        result = parallel(
            delayed(perform_fer)(path=path, mtcnn=mtcnn)
            for path in path_lst
        )
    
    return result


def split_players(emotions: list, width=1280):
    """Split emotions for two players given their position on screen.
    
    Parameters
    ----------
    emotions : list of dict
        Array of emotions info for each frame.
    width : int
        Video width to split screen in half.
        
    Returns
    -------
    left_player, right_player : dict
        Separate dictionary with emotions for each player.
    """
    emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
    left_player = defaultdict(lambda: [])
    right_player = defaultdict(lambda: [])
    
    for emotion in emotions:
        if emotion['box0'] is None:
#             left_player['box'].append(None)
#             right_player['box'].append(None)
            
            for emotion_name in emotion_names:
                left_player[emotion_name].append(None)
                right_player[emotion_name].append(None)
        else:
            if emotion['box0'][0] < (width/2):
#                 left_player['box'].append(emotion['box0'])
                for emotion_name in emotion_names:
                    left_player[emotion_name].append(emotion[emotion_name+'0'])
                    
                if 'box1' in emotion:
#                     right_player['box'].append(emotion['box1'])
                    for emotion_name in emotion_names:
                        right_player[emotion_name].append(emotion[emotion_name+'1'])
                else:
#                     right_player['box'].append(None)
                    for emotion_name in emotion_names:
                        right_player[emotion_name].append(None)
            else:
#                 right_player['box'].append(emotion['box0'])
                for emotion_name in emotion_names:
                    right_player[emotion_name].append(emotion[emotion_name+'0'])
                    
                if 'box1' in emotion:
#                     left_player['box'].append(emotion['box1'])
                    for emotion_name in emotion_names:
                        left_player[emotion_name].append(emotion[emotion_name+'1'])
                else:
#                     left_player['box'].append(None)
                    for emotion_name in emotion_names:
                        left_player[emotion_name].append(None)

    return dict(left_player), dict(right_player)