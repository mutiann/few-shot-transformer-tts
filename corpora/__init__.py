import os

dataset_path = os.path.expanduser(r'~/data/free/base')
transformed_path = os.path.expanduser(r'~/data/free/processed')
packed_path = os.path.expanduser(r'~/data/free/processed_mel')
# Containing metadata.csv, audiometa.json, and mel/, in dir of each dataset
# and lang_to_id.json, spk_id.json, in transformed_path

include_corpus = ['caito_de_de', 'caito_en_uk', 'caito_en_us', 'caito_es_es', 'caito_fr_fr',
                  'caito_it_it', 'caito_pl_pl', 'caito_ru_ru', 'caito_uk_ua',
                  'css10_de', 'css10_el', 'css10_es', 'css10_fi', 'css10_fr', 'css10_hu', 'css10_ja',
                  'css10_nl', 'css10_zh', 'css10_ru',
                  'lsru', 'jsut', 'kss', 'ljspeech', 'pt_br', 'siwis', 'thorsten', 'enbible',
                  'google_bn_bd', 'google_bn_in', 'google_eu_es', 'google_gl_es', 'google_gu_in',
                  'google_jv_id', 'google_km_kh', 'google_kn_in', 'google_ml_in', 'google_my_mm', 'google_ne_np',
                  'google_si_lk', 'google_su_id', 'google_ta_in', 'google_te_in',
                  'google_yo_ng', 'databaker', 'nst_da', 'nst_nb']

T1 = 'caito_ru_ru:lsru:css10_ru:caito_es_es:css10_es:caito_de_de:thorsten:css10_de:caito_en_us:ljspeech:enbible'
T2 = 'caito_fr_fr:siwis:css10_fr:caito_en_uk:caito_uk_ua:jsut:css10_ja:' \
     'css10_zh:databaker'
target = 'css10_el'
T3 = ':'.join(set(include_corpus).difference(T1.split(":") + T2.split(":") + target.split(':')))

dataset_language = {'css10_de': 'de_de', 'css10_el': 'el_gr', 'css10_es': 'es_es', 'css10_fi': 'fi_fi',
                    'css10_fr': 'fr_fr', 'css10_hu': 'hu_hu', 'css10_ja': 'ja_jp', 'css10_nl': 'nl_nl',
                    'css10_zh': 'zh_cn', 'css10_ru': 'ru_ru', 'lsru': 'ru_ru',
                    'jsut': 'ja_jp', 'kss': 'ko_kr', 'ljspeech': 'en_us', 'pt_br': 'pt_br', 'siwis': 'fr_fr',
                    'thorsten': 'de_de', 'databaker': 'zh_cn', 'enbible': 'en_us', 'nst_da': 'da_dk', 'nst_nb': 'nb_no'}

def get_dataset_language(dataset_name):
    if dataset_name.startswith('google') or dataset_name.startswith('caito'):
        return dataset_name[-5:]
    return dataset_language[dataset_name]
