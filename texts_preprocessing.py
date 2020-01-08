import pymorphy2
lemmatizer = pymorphy2.MorphAnalyzer()

from nltk.corpus import stopwords
stop_words = stopwords.words("russian") + stopwords.words('english')

import re
stop_pattern = re.compile(r'\b(' + r'|'.join(stop_words) + r')\b\s*')
def del_stop_words(text):
    return stop_pattern.sub('', text)

spec_pattern = re.compile(r'(' + r'|'.join(['</p>', '<p>', '</li>', '<li>', '</ul>',
                                            '<ul>', '<strong>', '</strong>']) + r')')
def del_spec_symbols(text):
    return spec_pattern.sub('', text)


def text_preprocess(text, keep_order=False):
    text = del_stop_words(text)
    text = del_spec_symbols(text)
    text = re.sub('\W+',' ', text).lower().strip()
    if keep_order:
        words = text.split(' ')
    else:
        words = set(text.split(' '))
    text = []
    for word in words:
        text.append(lemmatizer.parse(word)[0].normal_form)
    return ' '.join(text)
