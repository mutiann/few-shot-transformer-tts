import logging
import unicodedata
pad_id = 0
eos_id = 1
sos_id = 2

def is_sep(ch):
    if unicodedata.category(ch) in ["Pc", "Pd", "Pe", "Pf", "Pi", "Po", "Ps", "Zl", "Zp", "Zs"]:
        return True
    return False


def text_to_byte_sequence(text: str, use_sos=True, use_eos=True):
    s = list(text.encode('utf-8'))
    if use_sos:
        s = [sos_id] + s
    if use_eos:
        s = s + [eos_id]
    return s

def language_name_to_id(lang_to_id, lang):
    id_to_lang = dict([(v, k) for k, v in lang_to_id.items()])
    if isinstance(lang, str):
        lang = lang.split(':')
    else:
        lang = list(lang)
    for i in range(len(lang)):
        if lang[i].isnumeric():
            if lang[i] not in id_to_lang:
                logging.warn('Unknown language requested: ' + str(lang[i]))
        else:
            if lang[i] in lang_to_id:
                lang[i] = lang_to_id[lang[i]]
            else:
                logging.warn('Unknown language requested: ' + str(lang[i]))
    lang = [t for t in lang if t in id_to_lang]
    logging.info('Selected languages: ' + ' '.join([id_to_lang[t] for t in lang]))
    return lang

def language_vec_to_id(lv):
    for i in range(len(lv)):
        if lv[i] > 0:
            return i
    return -1