import os, re, random, json
import xml.etree.ElementTree as ET
from tqdm import tqdm
from datetime import datetime
from contextlib import contextmanager
from gensim.models import Word2Vec

# tokenizers
from konlpy.tag import Okt # korean
from somajo import SoMaJo # german
import fugashi # japanese
from camel_tools.tokenizers.word import simple_word_tokenize # arabic
from camel_tools.disambig.mle import MLEDisambiguator # arabic

@contextmanager
def time_(name):
    start_time = datetime.now()
    yield
    elapsed_time = datetime.now() - start_time
    print(f'{name} finished in {str(elapsed_time)[:7]}')

def create_corpus_word_kn():
    '''
    prepare corpus_word for korean news
    '''
    
    wikicorpus, corpus = 'wiki_ko', 'kn'
    
    okt = Okt()
    corpus_words = []

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')

    dirs = os.listdir('data/kn')
    for d in dirs:
        fns = os.listdir(f'data/kn/{d}')
        for fn in tqdm(fns):
            if fn[-6:-4] == 'ko':
                with open(f'data/kn/{d}/{fn}') as f:
                    obj = f.read()
                    paras = obj.split('\n\n')
                    for para in paras:
                        if len(para) > 30:
                            sents = re.split('[.,]', para)
                            sent_text = ''
                            for sent in sents:
                                tokens = [t for t in okt.morphs(sent, norm=True, stem=True) if re.match('^[가-힣]+$', t)] # korean char only
                                tokens = [t for t in tokens if t not in stopwords] # not stopword
                                if len(tokens) > 1:
                                    sent_text += ' '.join(tokens) + '\n' # '\n' to separate lines
                            corpus_words += [sent_text.strip()] # extra '\n' to separate paragraph 

    random.shuffle(corpus_words)
    print('number of paragraph:', len(corpus_words))

    with open(f'corpus/{corpus}/corpus_word.txt', 'w') as f:
        f.write('\n\n'.join(corpus_words))

def create_corpus_word_de():
    '''
    prepare corpus_word for german news
    '''
    
    wikicorpus, corpus = 'wiki_de', 'dn'

    tokenizer = SoMaJo("de_CMC", split_camel_case=True)
    corpus_words = []

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')

    paras = []
    with open('data/dn/articles.csv') as f:
        for line in f:
            paras += [line.strip().split(';')[1]]

    # tokenize a paragraph at a time
    for para in tqdm(paras): 
        sents = tokenizer.tokenize_text([para])
        sent_text = ''
        for sent in sents:
            tokens = [t.text for t in sent if re.match('^[a-zA-ZäöüÄÖÜß]+$', t.text)] # german char only
            tokens = [t for t in tokens if t not in stopwords]
            if len(tokens) > 1:
                sent_text += ' '.join(tokens) + '\n' # '\n' to separate lines
        corpus_words += [sent_text.strip()] # extra '\n' to separate paragraph 

    random.shuffle(corpus_words)
    print('number of paragraph:', len(corpus_words))

    with open(f'corpus/{corpus}/corpus_word.txt', 'w') as f:
        f.write('\n\n'.join(corpus_words))

def create_corpus_word_jn():
    '''
    prepare corpus_word for korean news
    '''
    
    wikicorpus, corpus = 'wiki_ja', 'jn'
    
    tagger = fugashi.Tagger()
    corpus_words = []

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')
    
    fns = os.listdir(f'data/jn')
    for fn in tqdm(fns):
        with open(f'data/jn/{fn}') as f:
            obj = json.load(f)
            para = obj['text']
            sents = re.split('[。、]', para)
            sent_text = ''
            for sent in sents:
                # japanese char only # split('-')[0] because of katakana
                tokens = [word.feature.lemma.split('-')[0] for word in tagger(sent) if re.match('^[一-龠ぁ-ゔァ-ヴー]+$', word.surface) and word.feature.lemma!=None] 
                tokens = [t for t in tokens if t not in stopwords] # not stopword
                if len(tokens) > 1:
                    sent_text += ' '.join(tokens) + '\n' # '\n' to separate lines
            corpus_words += [sent_text.strip()] # extra '\n' to separate paragraph 

    random.shuffle(corpus_words)
    print('number of paragraph:', len(corpus_words))

    with open(f'corpus/{corpus}/corpus_word.txt', 'w') as f:
        f.write('\n\n'.join(corpus_words))

def create_corpus_word_an():
    '''
    prepare corpus_word for korean news
    '''
    
    wikicorpus, corpus = 'wiki_ar', 'an'
    
    mle = MLEDisambiguator.pretrained()
    corpus_words = []

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')

    dirs = os.listdir(f'data/{corpus}')
    count = 0
    for d in dirs:
        fns = os.listdir(f'data/{corpus}/{d}')
        for fn in tqdm(fns):
            try:
                tree = ET.parse(f'data/{corpus}/{d}/{fn}')
            except:
                continue
            root = tree.getroot()
            phrase = root.find('./TEXT')
            for line in phrase.text.split('\n'):
                sents = re.split('[.]', line)
                sent_text = ''
                for sent in sents:
                    sentence = simple_word_tokenize(sent)
                    disambig = mle.disambiguate(sentence)
                    try:
                        tokens = [d.analyses[0].analysis['lex'] for d in disambig]
                        tokens = [t for t in tokens if re.match('^[\u0600-\u06FF]+$', t)]  
                        tokens = [t for t in tokens if t not in stopwords] # not stopword
                        if len(tokens) > 1:
                            sent_text += ' '.join(tokens) + '\n' # '\n' to separate lines
                    except:
                        count += 1
                
                if len(sent_text.strip()) > 0:
                    corpus_words += [sent_text.strip()] # extra '\n' to separate paragraph 
    
    if count > 0:
        print('n_errors', count)

    random.shuffle(corpus_words)
    print('number of paragraph:', len(corpus_words))

    with open(f'corpus/{corpus}/corpus_word.txt', 'w') as f:
        f.write('\n\n'.join(corpus_words))

def prepare_dir(corpus, type_):
    '''
    remove previous directory if existed,
    and create train and test directories each type of collocation
    '''
    os.system(f'rm corpus/{corpus}/{type_} -r -f')
    os.system(f'mkdir corpus/{corpus}/{type_}')
    os.system(f'mkdir corpus/{corpus}/{type_}/dir_test')
    os.system(f'mkdir corpus/{corpus}/{type_}/dir_train')

def write_train_test(corpus, type_, paras):
    '''
    write data from paras into files that mallet needs
    '''
    prepare_dir(corpus, type_)

    # test
    for i, para in enumerate(paras[:len(paras)//4]): # first 25%
        with open(f'corpus/{corpus}/{type_}/dir_test/{i}.txt', 'w') as f:
            f.write(para.replace('\n', ' '))
    # train
    for i, para in enumerate(paras[len(paras)//4:]): # last 75%
        with open(f'corpus/{corpus}/{type_}/dir_train/{i}.txt', 'w') as f:
            f.write(para.replace('\n', ' '))

def prepare_mallet_data(corpus, types):
    '''
    prepare files that mallet needs
    '''
    for type_ in types:
        prepare_dir(corpus, type_)
        with open(f'corpus/{corpus}/corpus_{type_}.txt') as f:
            paras = f.read().split('\n\n')
        write_train_test(corpus, type_, paras)                                          

def merge_tokens(text, top_bigrams):
    '''
    merge text using top bigrams
    '''
    tokens = text.strip().split()
    merged_text = tokens[0] # text starts with first token

    for i in range(len(tokens)-1):
        bigram = tokens[i]+'\t'+tokens[i+1]
        if bigram in top_bigrams:
            merged_text += '_' + tokens[i+1] # merge
        else:
            merged_text += ' ' + tokens[i+1] # not merge
    
    return merged_text

def create_corpus(corpus, wikicorpus, types):
    '''
    create corpus for collocations
    '''
    for type_ in types:

        corpus_text = ''
        # load bi_to_vocabs
        with open(f'corpus/{wikicorpus}/top_bigrams_{type_}.txt') as f:
            top_bigrams = f.read().split('\n')
            
        # load original text
        with open(f'corpus/{corpus}/corpus_word.txt') as f:
            paras = f.read().split('\n\n')
        
        # merge tokens
        for para in tqdm(paras):
            sents = para.split('\n')
            for sent in sents:
                if sent.strip().split() != []:
                    corpus_text += merge_tokens(sent, top_bigrams) + '\n' # merge tokens
            corpus_text += '\n'
        
        # write files
        with open(f'corpus/{corpus}/corpus_{type_}.txt', 'w') as f:
            f.write(corpus_text.strip())

def prepare_w2v_input(corpus, types):
    '''
    open corpus for each type,
    convert to list of list of tokens, and
    save as input for word2vec
    '''
    input_ = []
    for type_ in types:
        with open(f'corpus/{corpus}/corpus_{type_}.txt') as f:
            paras = f.read().split('\n\n')
            sents = [sent.split() for para in paras for sent in para.split('\n')]
            input_ += sents
    return input_

def train_w2v(corpus, types, vector_size=100, min_count=1, epochs=40):
    input_ = prepare_w2v_input(corpus, types)
    with time_('w2v'):
        model = Word2Vec(input_, vector_size=vector_size, min_count=min_count, epochs=epochs)
        model.save(f'corpus/{corpus}/w2v_model')



