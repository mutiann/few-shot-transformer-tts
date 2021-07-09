import os
import logging
import editdistance
import re
import unicodedata
import json
import traceback
import requests

if os.path.exists("azure_key.json"):
    config = json.load(open("azure_key.json", 'r'))
    transcribe_available = True
else:
    transcribe_available = False

def basic_normalize(text, locale):
    text_ = ''
    for ch in text:
        if unicodedata.category(ch) in ["Pc", "Pd", "Ps", "Pe", "Pi", "Pf", "Po"]:
            continue
        if locale in ['zh', 'zh-cn', 'th-th', 'zh-tw', 'zh-hk', 'ja-jp', 'ko-kr'] and ch == ' ':
            continue
        text_ += ch.lower()
    text_ = re.sub(r'\s+', ' ', text_)
    text_ = unicodedata.normalize('NFD', text_)
    return text_.strip()


def azure_transcribe(audio_path, lang):
    if lang == 'zh':
        lang = 'zh-cn'
    endpoint = r"https://%s.stt.speech.microsoft.com/speech/recognition/conversation/cognitiveservices/v1?" \
               r"format=detailed&profanity=raw&language=%s" % (config['region'], lang)
    header = {'Ocp-Apim-Subscription-Key': config['subscription'], 'Content-Type': 'audio/wav'}
    data = open(audio_path, 'rb').read()
    response = requests.post(endpoint, data=data, headers=header)
    if response.status_code != 200:
        return None
    result = json.loads(response.content)
    return result


def transcribe(wav_path, meta, id_to_lang):
    lang = id_to_lang(meta['i'])
    for i in range(5):  # Try 5 times
        try:
            assert os.path.exists(wav_path), wav_path + ' not exists'
            result = azure_transcribe(wav_path, lang)
            if result is None or result['RecognitionStatus'] != 'Success':
                raise ValueError("Fail to transcribe " + str(result))
            result['locale'] = lang
            result['name'] = meta['n'][:-4]
            result['truth'] = truth = basic_normalize(meta['t'], lang)
            result['pred'] = pred = basic_normalize(result['NBest'][0]['Lexical'], lang)
            cer = min(1.0, editdistance.eval(truth, pred) / (len(pred) + 1e-9))
            logging.info('%s %.3f: "%s" | "%s"' % (
                result['name'], cer, truth.encode('unicode-escape'), pred.encode('unicode-escape')))
            result['cer'] = cer
            return result
        except:
            logging.error('Fail to transcribe %s, retry... (%s)' % (wav_path, meta))
            logging.error(traceback.format_exc())
    return {'cer': 1.0, 'locale': lang, 'name': meta['n'][:-4], 'DisplayText': '', 'fail': True}
