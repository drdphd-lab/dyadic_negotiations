from copy import deepcopy
from functools import reduce
import sys
sys.path.append('/Users/vall/opt/anaconda3/envs/stt_env/lib/python3.9/site-packages/')

from huggingsound import SpeechRecognitionModel
from joblib import Parallel, delayed
from pyannote.audio import Pipeline


SMALL_MODEL = "jonatasgrosman/wav2vec2-large-xlsr-53-russian"
BIG_MODEL = "jonatasgrosman/wav2vec2-xls-r-1b-russian"


def diarize_audio(path: str):
    """Extract speakers from a given audio.
    
    Parameters
    ----------
    path : str
        Path to audio file.
        
    Returns
    -------
    diarization_lst : list of dict
        Information about start, stop and speaker turn.
    
    Notes
    -----
    Model is pretty slow. Might take 4-5 minutes to process 30-second audio.
    Start and stop are given in seconds.
    """
    model = Pipeline.from_pretrained("pyannote/speaker-diarization")
    output = model(path)
    
    diarization_lst = []
    for turn, _, speaker in output.itertracks(yield_label=True):
        temp_dct = {}
        temp_dct['start'] = turn.start
        temp_dct['stop'] = turn.end
        temp_dct['speaker'] = speaker
        diarization_lst.append(temp_dct)
        
    return diarization_lst


def diarize_audio_parallel(path_lst: list, **parallel_kwargs):
    """Extract speakers from multiple audios in parallel.
    
    Parameters
    ----------
    path_lst : list of str
        Paths to audio files.
    **parallel_kwargs : dict, optional
        Keyword arguments for joblib.Parallel class.
        
    Returns
    -------
    result : list of list
        Information about start, stop and speaker turn.
    """
    with Parallel(**parallel_kwargs) as parallel:
        result = parallel(
            delayed(diarize_audio)(path=path)
            for path in path_lst
        )
    
    return result


def transcribe_audio(path: str, model=BIG_MODEL):
    """Transcribe audio file.
    
    Parameters
    ----------
    path : str
        Path to audio file.
    model : str
        Model to use. Can be any from HuggingFace Hub.
        
    Returns
    -------
    transcription : dict
        Dictionary with speech-to-text, timestamps for each word and their probabilities.
        
    Notes
    -----
    Despite its size model is pretty fast. Takes about 45 seconds to process 30-second audio.
    Timestamp are given in miliseconds. Divide by 1000 to convert to seconds.
    """
    stt_model = SpeechRecognitionModel(model)
    transcription = stt_model.transcribe([path])
    
    return transcription[0]


def transcribe_audio_parallel(path_lst: list, **parallel_kwargs):
    """Transcribe multiple audio files in parallel.
    
    Parameters
    ----------
    path_lst : list
        Paths to audio files.
    **parallel_kwargs : dict, optional
        Keyword arguments for joblib.Parallel class.
    
    Returns
    -------
    result : list of dict
        Words, theirs timestamps and probabilities.
    
    Notes
    -----
    Model requires a lot of RAM. So be careful with n_jobs.
    For example, 3 jobs ate 25+ GB of RAM when using big model.
    """
    with Parallel(**parallel_kwargs) as parallel:
        result = parallel(
            delayed(transcribe_audio)(path=path)
            for path in path_lst
        )
    
    return result


def collapse_to_dct(stt_result: list):
    """Collapse speech to text parallel result to one dictionary."""
    stt = deepcopy(stt_result)
    if not stt[-1]['transcription']:
        stt.pop()  # remove last empty element
    
    final_dct = {}
    final_dct['transcription'] = ' '.join(map(lambda x: x['transcription'], stt))
    
    final_dct['start_timestamps'] = list(
        reduce(
            lambda a, b: a + list(map(lambda x: x + a[-1], b)), 
            map(lambda x: x['start_timestamps'], stt)
        )
    )
    
    final_dct['end_timestamps'] = list(
        reduce(
            lambda a, b: a + list(map(lambda x: x + a[-1], b)), 
            map(lambda x: x['end_timestamps'], stt)
        )
    )
    
    return final_dct