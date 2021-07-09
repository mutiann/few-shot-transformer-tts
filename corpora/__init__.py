import os

dataset_path = r"D:\free_corpus\base"
transformed_path = r"D:\free_corpus\processed"
packed_path = r"D:\free_corpus\packed"
# Containing metadata.csv and mels/, in dir of each dataset
# and lang_to_id.json and spk_id.json in transformed_path and packed_path

include_corpus = ['caito_de_de', 'caito_en_uk', 'caito_en_us', 'caito_es_es', 'caito_fr_fr',
                  'caito_it_it', 'caito_pl_pl', 'caito_ru_ru', 'caito_uk_ua',
                  'css10_de', 'css10_el', 'css10_es', 'css10_fi', 'css10_fr', 'css10_hu', 'css10_ja',
                  'css10_nl', 'css10_zh', 'css10_ru', 'databaker', 'enbible',
                  'google_bn_bd', 'google_bn_in', 'google_ca_es', 'google_eu_es', 'google_gl_es', 'google_gu_in',
                  'google_jv_id', 'google_km_kh', 'google_kn_in', 'google_ml_in', 'google_mr_in', 'google_my_mm',
                  'google_ne_np', 'google_si_lk', 'google_su_id', 'google_ta_in', 'google_te_in', 'google_yo_ng',
                  'jsut', 'kss', 'ljspeech', 'lsru', 'nst_da', 'nst_nb', 'pt_br', 'siwis', 'thorsten',
                  'hifi_us', 'hifi_uk', 'rss']

dataset_language = {'css10_de': 'de-de', 'css10_el': 'el-gr', 'css10_es': 'es-es', 'css10_fi': 'fi-fi',
                    'css10_fr': 'fr-fr', 'css10_hu': 'hu-hu', 'css10_ja': 'ja-jp', 'css10_nl': 'nl-nl',
                    'css10_zh': 'zh-cn', 'css10_ru': 'ru-ru', 'lsru': 'ru-ru',
                    'jsut': 'ja-jp', 'kss': 'ko-kr', 'ljspeech': 'en-us', 'pt_br': 'pt-br', 'siwis': 'fr-fr',
                    'thorsten': 'de-de', 'databaker': 'zh-cn', 'enbible': 'en-us', 'nst_da': 'da-dk', 'nst_nb': 'nb-no',
                    'hifi_us': 'en-us', 'hifi_uk': 'en-uk', 'rss': 'ro-ro'}

def get_dataset_language(dataset_name):
    if dataset_name.startswith('google') or dataset_name.startswith('caito'):
        return dataset_name[-5:].replace('_', '-')
    return dataset_language[dataset_name]
