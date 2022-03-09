import re, os, nltk

from bs4 import BeautifulSoup
from tqdm import tqdm
from random import sample

# custom
from util.util import time_

# tokenizers
from konlpy.tag import Okt # korean
from somajo import SoMaJo # german
import fugashi # japanese
from camel_tools.tokenizers.word import simple_word_tokenize # arabic
from camel_tools.disambig.mle import MLEDisambiguator # arabic
from attacut import tokenize # thai
from nltk.tokenize import sent_tokenize # english
from nltk.tokenize import word_tokenize # english

def create_wikidump_text(dir_):
    '''
    from the directory of extracted wikidump, create a text file
    '''
    texts = []    

    dirs = os.listdir(f'data/wikidump/{dir_}')
    for d in tqdm(dirs):
        fns = os.listdir(f'data/wikidump/{dir_}/{d}')
        for fn in fns:
            with open(f'data/wikidump/{dir_}/{d}/{fn}') as f:
                contents = f.read()
                soup = BeautifulSoup(contents, 'lxml')
                finds = soup.find_all('doc')
                for find in finds:
                    text = find.text.strip()
                    text = re.sub('\n+', '\n', text)
                    texts += text.split('\n')
                    
    with open(f'data/wikidump/{dir_}_text.txt', 'w') as f:
        f.write('\n'.join(texts))
        print(f'number of lines: {len(texts)}')

def reduce_wikidump(wikicorpus, factor):
    os.system(f'mv data/wikidump/{wikicorpus}_text.txt data/wikidump/{wikicorpus}_text_large.txt')
    with open(f'data/wikidump/{wikicorpus}_text_large.txt') as f:
        lines = f.read().strip().split('\n')
        smalls = sample(lines, len(lines)//factor )
    with open(f'data/wikidump/{wikicorpus}_text.txt', 'w') as f:
        f.write('\n'.join(smalls))

def create_corpus_batch(wikicorpus, begin=0, end=10):
    '''
    create a batche of wpe corpus,
    be careful, sentences separators
    '''
    with open(f'data/wikidump/{wikicorpus}_text.txt') as f:
        lines = f.read().split('\n')

    with open(f'corpus/{wikicorpus}/stopwords.txt') as f:
        stopwords = f.read().strip().split('\n')

    n_lines = len(lines) // 10 # Number of line per batch

    # tokenizer
    if wikicorpus == 'wiki_ko':
        okt = Okt()
    elif wikicorpus == 'wiki_de':
        tokenizer = SoMaJo("de_CMC", split_camel_case=True)
    elif wikicorpus == 'wiki_ja':
        tagger = fugashi.Tagger()
    elif wikicorpus == 'wiki_ar':
        mle = MLEDisambiguator.pretrained()
    
    for i in range(begin, end):
        ls = lines[i * n_lines: (i + 1) * n_lines]
        text = ''    

        if wikicorpus == 'wiki_ko':
            for l in tqdm(ls): 
                sents = re.split('[.,]', l)
                for sent in sents:
                    tokens = [t for t in okt.morphs(sent, norm=True, stem=True) if re.match('^[가-힣]+$', t)] # korean char only
                    tokens = [t for t in tokens if t not in stopwords] # not stopword
                    if len(tokens) > 1:
                        text += ' '.join(tokens) + '\n'
        
        elif wikicorpus == 'wiki_de':
            for para in tqdm(ls): 
                sents = tokenizer.tokenize_text([para])
                for sent in sents:
                    tokens = [t.text for t in sent if re.match('^[a-zA-ZäöüÄÖÜß]+$', t.text)] # german char only
                    tokens = [t for t in tokens if t not in stopwords]
                    if len(tokens) > 1:
                        text += ' '.join(tokens) + '\n'

        elif wikicorpus == 'wiki_ja':
            for l in tqdm(ls): 
                sents = re.split('[。、]', l)
                for sent in sents:
                    # japanese char only # split('-')[0] because of katakana
                    tokens = [word.feature.lemma.split('-')[0] for word in tagger(sent) if re.match('^[一-龠ぁ-ゔァ-ヴー]+$', word.surface) and word.feature.lemma!=None]  
                    tokens = [t for t in tokens if t not in stopwords] # not stopword
                    if len(tokens) > 1:
                        text += ' '.join(tokens) + '\n'

        elif wikicorpus == 'wiki_ar':
            count = 0
            for l in tqdm(ls): 
                sents = re.split('[.]', l)
                for sent in sents:
                    sentence = simple_word_tokenize(sent)
                    disambig = mle.disambiguate(sentence)
                    try:
                        tokens = [d.analyses[0].analysis['lex'] for d in disambig]
                        tokens = [t for t in tokens if re.match('^[\u0600-\u06FF]+$', t)]  
                        tokens = [t for t in tokens if t not in stopwords] # not stopword
                        if len(tokens) > 1:
                            text += ' '.join(tokens) + '\n'
                    except:
                        count += 1
            print('n_errors', count)

        elif wikicorpus == 'wiki_th':
            for l in tqdm(ls): 
                tokens = tokenize(l)
                tokens = [t for t in tokens if re.match('^[ก-์]+$', t)] # thai word only
                tokens = [t for t in tokens if t not in stopwords] # not stopword
                if len(tokens) > 1:
                    text += ' '.join(tokens) + '\n'

        elif wikicorpus == 'wiki_cn':
            for l in tqdm(ls): 
                sents = re.split('[。]', l)
                for sent in sents:
                    tokens = [t for t in sent.strip().split() if re.match('^[\u4e00-\u9fa5]+$', t)] # chinese char only
                    tokens = [t for t in tokens if t not in stopwords] # not stopword
                    if len(tokens) > 1:
                        text += ' '.join(tokens) + '\n'

        elif wikicorpus == 'wiki_en':
            for l in tqdm(ls): 
                sents = sent_tokenize(l)
                for sent in sents:
                    tokens = [t for t in word_tokenize(sent) if re.match('^[a-zA-Z]+$', t)] # english word only
                    tokens = [t for t in tokens if t not in stopwords]
                    if len(tokens) > 1:
                        text += ' '.join(tokens) + '\n'


        output_path = f'corpus/{wikicorpus}/corpus_{i}.txt'
        with open(output_path, 'w') as f:
            f.write(text)

def create_corpus_combind(dir_):
    '''
    combind corpus_i.txt into corpus.txt
    '''
    text = ''
    # read each file
    for i in range(10):
        with open(f'corpus/{dir_}/corpus_{i}.txt') as f:
            obj = f.read()
            text += obj + '\n'
    
    # write final file
    with open(f'corpus/{dir_}/corpus.txt', 'w') as f:
        f.write(text.strip()) # remove last '\n'

def create_top_bigrams(wikicorpus, n_bigrams):
    # get tokens from corpus
    with open(f'corpus/{wikicorpus}/corpus.txt') as f:
        corpus = f.read().split('\n')
        tokens = [token for line in corpus for token in line.split()]

    # find bigrams
    with time_('bigram'):
        bigrams = nltk.collocations.BigramAssocMeasures()
        bigramFinder = nltk.collocations.BigramCollocationFinder.from_words(tokens)

    for type_ in ['freq', 't', 'chi']:
        with time_(type_):
            if type_ == 'chi':
                bigramTable = bigramFinder.score_ngrams(bigrams.chi_sq)
            elif type_ == 't':
                bigramTable = bigramFinder.score_ngrams(bigrams.student_t)
            elif type_ == 'freq':
                bigramTable = bigramFinder.score_ngrams(bigrams.raw_freq)

            texts = ['\t'.join(bigram) for bigram, _ in bigramTable[:n_bigrams]]
            with open(f'corpus/{wikicorpus}/top_bigrams_{type_}.txt', 'w') as f:
                f.write('\n'.join(texts))



