import torch

import pandas as pd

from pyaspeller import YandexSpeller

from natasha import (
    Segmenter,
    MorphVocab,
    
    NewsEmbedding,
    NewsMorphTagger,

    Doc
)

VALID_POS = ['VERB', 'NOUN', 'ADJ', 'ADV']


def restore_punctuation(text: list, lan='ru'):
    """Restore punctuation.
    
    Parameters
    ----------
    text : array-like of str
        Text without any punctuation.
    lan : {'ru', 'en', 'de', 'es'}
        Text language.
        
    Returns
    -------
    enhanced_text : array-like of str
        Text with restored punctuation.
    """
    model, example_texts, languages, punct, apply_te = torch.hub.load(
        repo_or_dir='snakers4/silero-models', model='silero_te'
    )
    
    enhanced_text = []
    for line in text:
        lower_line = line.lower()
        enhanced_line = apply_te(lower_line, lan=lan)
        enhanced_text.append(enhanced_line)
        
    return enhanced_text


def spell_text(text: list):
    """Correct spelling errors.
    
    Parameters
    ----------
    text : array-like of str
        Text with spelling errors.
        
    Returns
    -------
    fixed_text : array-like of str
        Text with fixed spelling errors.
    """
    speller = YandexSpeller()
    
    fixed_text = []
    for line in text:
        fixed_line = speller.spelled(line)
        fixed_text.append(fixed_line)
        
    return fixed_text


def coalesce_speaker(diarization):
    """Connect adjacent intervals for one speaker.
    
    Parameters
    ----------
    diarization : list of dict
        Array with diarization information for a certain audio.
        Each dict has 'start', 'stop' and 'speaker' as keys.
    
    Returns
    -------
    new_diarization : list of dict
        Diarization with connected intervals
        
    Notes
    -----
    If a speaker talks for some time and then keeps silent for a moment
    voice activity detector (VAD) cathes it and divides speaking turns though
    speaker doesn't change.
    This function helps fill those moments of silence and convert different
    speech intervals (belonging to same person) into one.
    """
    new_diarization = [
        {
            'start': diarization[0]['start'],
            'stop': diarization[0]['stop'],
            'speaker': diarization[0]['speaker']
        }
    ]
    current_speaker = diarization[0]['speaker']
    
    for turn in diarization:
        if current_speaker == turn['speaker']:
            new_diarization[-1]['stop'] = turn['stop']
        else:
            new_diarization.append(
                {
                    'start': turn['start'],
                    'stop': turn['stop'],
                    'speaker': turn['speaker']
                }
            )
            current_speaker = turn['speaker']
            
    return new_diarization


def combine_speech_and_speakers(captions, diarization):
    """Combine captions and diarization.
    
    Parameters
    ----------
    captions : list of dict
        Array with captions for a certain audio/video file.
        Each dict has 'text', 'start' and 'duration' as keys.
    diarization : list of dict
        Array with diarization information for a certain audio.
        Each dict has 'start', 'stop' and 'speaker' as keys.
    
    Returns
    -------
    speech_df : pandas DataFrame
        Table with speech and its time.
        
    Notes
    -----
    As it's impossible to perfectly match YouTube captions time intervals
    and diarization time intervals the following algorithm simply looks if
    diarization interval includes YouTube interval.
    If there are unused captions (i.e. they were skipped) they are added
    to the latest speaker and ignored if IndexError raises.
    """
    speech_dct = {'text': [], 'speaker': [], 'start': [], 'stop': []}
    last_used_caption = 0
    
    for turn in diarization:
        speech_dct['start'].append(turn['start'])
        speech_dct['stop'].append(turn['stop'])
        speech_dct['speaker'].append(turn['speaker'])
        
        temp_speech = []
        for idx, caption in enumerate(captions):            
            condition = (
                (caption['start'] >= turn['start']) &
                (caption['start'] + caption['duration'] <= turn['stop'])
            )
            if condition:
                if idx > last_used_caption + 1:
                    for unused_caption in range(last_used_caption + 1, idx):
                        try:
                            speech_dct['text'][-1] += f' {captions[unused_caption]["text"]}'
                        except IndexError:
                            pass
                
                temp_speech.append(caption['text'])
                last_used_caption = idx
        
        speech_dct['text'].append(' '.join(temp_speech))
    
    speech_df = pd.DataFrame(speech_dct)
    
    return speech_df


def extract_judge_speech(captions, time_tuple):
    """Extract judge speech from captions.
    
    Parameters
    ----------
    captions : list of dict
        Caption for certain negotiations.
    time_tuple : array-like of str
        Array with 3 values of time when judges speak.
        
    Returns
    -------
    enhanced_judge_1, enhanced_judge_2, enhanced_judge_3 : str
        Enhanced (restored) speech for judges.
    """
    judge_1, judge_2, judge_3 = [], [], []
    time_to_int = list(map(lambda x: int(x[:2])*60 + int(x[3:]), time_tuple))
    
    for caption in captions:
        if (caption['start'] >= time_to_int[0]) & (caption['start'] < time_to_int[1]):
            judge_1.append(caption['text'])
        
        if (caption['start'] >= time_to_int[1]) & (caption['start'] < time_to_int[2]):
            judge_2.append(caption['text'])
        
        if caption['start'] >= time_to_int[2]:
            judge_3.append(caption['text'])
    
    enhanced_judge_1 = restore_punctuation([' '.join(judge_1).replace('.', '')])  # glitched dot
    enhanced_judge_2 = restore_punctuation([' '.join(judge_2).replace('.', '')])  # glitched dot
    enhanced_judge_3 = restore_punctuation([' '.join(judge_3).replace('.', '')])  # glitched dot
            
    return enhanced_judge_1[0], enhanced_judge_2[0], enhanced_judge_3[0]


def apply_natasha(text, valid_pos=VALID_POS):
    """Lemmatize text and leave words with only valid part of speech (POS)
    
    Parameters
    ----------
    text : str
        Text with sentences.
    valid_pos : list of str
        Valid parts of speech.
    
    Returns
    -------
    new_text : list of str
        Filtered text divided into sentences.
    """
    segmenter = Segmenter()
    morph_vocab = MorphVocab()
    emb = NewsEmbedding()
    morph_tagger = NewsMorphTagger(emb)

    new_text = []
    
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)
    
    for sent in doc.sents:
        new_sent = []
        
        for token in sent.tokens:
            token.lemmatize(morph_vocab)
            if token.pos in valid_pos:
                new_sent.append(token.lemma)
        
        if len(new_sent) >= 2:
            new_text.append(' '.join(new_sent))
            
    return new_text