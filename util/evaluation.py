import math, json
import xml.etree.ElementTree as ET
import pandas as pd

from collections import Counter
from pathlib import Path
from tqdm import tqdm
from gensim.models import Word2Vec
from scipy import spatial
from IPython.display import display

def add_train_info(corpora, types):
    '''
    count token and calculate log-likehood unigram of token
    '''
    for corpus in corpora:

        # create places in results.json
        with open('corpus/results.json') as f:
            results = json.load(f)
        try:
            results[corpus]
        except:
            results[corpus] = {}    
            with open('corpus/results.json', 'w') as f:
                json.dump(results, f)

        for type_ in types:
            tokens = []
            with open(f'corpus/{corpus}/corpus_{type_}.txt') as f:
                paras = f.read().split('\n\n')
                for para in paras[len(paras)//4:]: # train paras
                    tokens += para.replace('\n', ' ').split()
            tc = Counter(tokens) # counter

            # info form train data that is used to compute log likelihood of unigram for test data
            N, W, beta = len(tokens), len(tc), 0.01
            ll_token = {}
            for k,v in tc.items():
                ll_token[k] = math.log((v + beta) / (N + W * beta)) # base 10 # L_unigram formula page 290
            ll_token['unk'] = math.log((0 + beta) / (N + W * beta))
            with open(f'corpus/{corpus}/{type_}/ll_token.json', 'w') as f:
                json.dump(ll_token, f)

            # create places in results.json
            with open('corpus/results.json') as f:
                results = json.load(f)
            try:
                results[corpus][type_]
            except:
                results[corpus][type_] = {}
                with open('corpus/results.json', 'w') as f:
                    json.dump(results, f)

def add_avg_ll(corpora, types, n_topics, rounds):
    '''
    get log-likelihood from mallet
    then compute normalized log-likelihood per token by subtracting ll unigram per token
    '''
    for corpus in corpora:
        for type_ in types:
            for n_topic in n_topics:

                # load ll_token
                with open(f'corpus/{corpus}/{type_}/ll_token.json') as f:
                    ll_token = json.load(f)

                # compute likelihood of unigram for test data
                tokens = []
                with open(f'corpus/{corpus}/corpus_{type_}.txt') as f:
                    paras = f.read().split('\n\n')
                    for para in paras[:len(paras)//4]: # train paras
                        tokens += para.replace('\n', ' ').split()

                ll_unigram = sum([ll_token.get(token, ll_token['unk']) for token in tokens])
                n_token_test = len(tokens)
                ll_unigram_per_token = ll_unigram / n_token_test

                # get average ll from prob and compute ll_per_token
                lls = []
                for r in rounds:
                    with open(f'corpus/{corpus}/{type_}/{n_topic}/{r}/prob') as file:
                        obj = file.read()
                        lls.append(float(obj))

                ll = sum(lls) / len(rounds) # 10 is number of round
                ll_per_token = ll / n_token_test
                ll_per_token_norm = ll_per_token - ll_unigram_per_token

                # get cohereance from diagnostics.xml and compute average coherence
                avg_coherences = []
                for r in rounds:
                    doc = ET.parse(f'corpus/{corpus}/{type_}/{n_topic}/{r}/diagnostics.xml')
                    root = doc.getroot()
                    coherences = [float(topic.get('coherence')) for topic in root.findall('topic')]
                    avg_coherences.append(sum(coherences) / len(coherences))
                avg_coherence = sum(avg_coherences) / len(rounds) # 10 is number of round

                # update results
                with open ('corpus/results.json') as f:
                    results = json.load(f)
                # create an empty dict if it is not there
                try:
                    results[corpus][type_][str(n_topic)]
                except:
                    results[corpus][type_][str(n_topic)] = {}
                # update 
                results[corpus][type_][str(n_topic)]['ll_per_token_norm'] = ll_per_token_norm
                results[corpus][type_][str(n_topic)]['avg_coherence'] = avg_coherence
                # write results
                with open ('corpus/results.json', 'w') as f:
                    json.dump(results, f)

def add_merged_percentage(corpora, types):
    '''
    count merged tokens and compute percentage
    '''
    with open('corpus/results.json') as f:
        results = json.load(f)
    
    for corpus in corpora:
        for type_ in types:
            if type_ == 'word':
                results[corpus][type_]['merged'] = 0
                continue
            with open(f'corpus/{corpus}/corpus_{type_}.txt') as f:
                obj = f.read()
                obj = obj.replace('\n\n', '\n').replace('\n', ' ')
                tokens = obj.split()
                merged_tokens = [t for t in tokens if '_' in t]  
                results[corpus][type_]['merged'] = len(merged_tokens) / len(tokens)
            print(f'{corpus:4s}{type_:5s}{len(merged_tokens) / len(tokens):.4f}')

    with open('corpus/results.json', 'w') as f:
        json.dump(results, f)

def get_keys(corpus, type_, n_topic, round, n_key):
    '''
    get keys which is list of list
    to be used to calculate silhouette
    '''
    with open(f'corpus/{corpus}/{type_}/{n_topic}/{round}/keys.txt') as f:
        lines = f.read().strip().split('\n')
        keys = [line.split('\t')[2] for line in lines]
        keys = [key.strip().split() for key in keys] # key is list of list
        keys = [[k for k in key if k not in ['_', '.', '【', '】']][:n_key] for key in keys] # filter out '.', '_', and want only n_key
        keys = [[k for k in key if k[0] != '_' and k[-1] != '_'] for key in keys]
        return keys

def calculate_silhouette(corpus, type_, n_topic, n_key, rounds):
    '''
    computing average silhouette 
    which is actually average (among rounds) of average (among topics) of silhouette 
    '''
    w2vmodel = Word2Vec.load(f'corpus/{corpus}/w2v_model') # load word2vec model
    silhouettes = []

    for r in tqdm(rounds):

        avg_s_list = []
        keys = get_keys(corpus, type_, n_topic, r, n_key)

        for topic in range(n_topic): # first topic

            thiskeys = keys[topic] # first topic
            aa, bb, ss = [], [], []

            for thiskey in thiskeys:
                # find a
                a = sum([spatial.distance.cosine(w2vmodel.wv.word_vec(j), w2vmodel.wv.word_vec(thiskey)) for j in thiskeys if j != thiskey]) / (n_key-1)
                aa.append(a)

                # find b
                avgs = []
                for t in range(n_topic):
                    if t != topic:
                        avg_distance = sum([spatial.distance.cosine(w2vmodel.wv.word_vec(k), w2vmodel.wv.word_vec(thiskey)) for k in keys[t]]) / n_key
                        avgs.append(avg_distance)
                b = min(avgs)
                bb.append(b)

                # find s
                s = (b-a)/max([b,a])
                ss.append(s)

            avg_s = sum(ss) / len(ss)
            avg_s_list.append(avg_s)

        silhouettes.append(sum(avg_s_list) / len(avg_s_list))  # avg_avg_s

    return sum(silhouettes) / len(rounds) # 10 is number of round

def add_silhouette(corpora, types, n_topics, n_keys, rounds):
    '''
    add average silhouette to results.json
    '''
    for corpus in corpora:
        print(f'corpus: {corpus}, add silhouette')
    
        for type_ in types:
            for n_topic in n_topics:
                for n_key in n_keys:
                    
                    silhouette = calculate_silhouette(corpus, type_, n_topic, n_key, rounds)

                    with open('corpus/results.json') as f:
                        results = json.load(f)
                    # add empty dict if needed
                    try:
                        results[corpus][type_][str(n_topic)][str(n_key)]
                    except:
                        results[corpus][type_][str(n_topic)][str(n_key)] = {}
                    
                    results[corpus][type_][str(n_topic)][str(n_key)]['silhouette'] = f"{silhouette:.4f}"    
                    
                    with open(Path('corpus/results.json'), 'w') as f:
                        json.dump(results, f) 

def export_results(corpora, types, n_topics, n_key):
    
    with open('corpus/results.json') as f:
        results = json.load(f)

    with open('corpus/abv_text.txt') as f:
        abvs = f.read().strip().split('\n')
        abv_text = {abv.split(':')[0]:abv.split(':')[1] for abv in abvs}
    
    # output Norm Log-Likehood
    output = ''
    line = f'Norm Log-Likehood,' # heads
    for n_topic in n_topics:
        for type_ in types:
            line += f'{abv_text[type_]}-{n_topic},'
    output += line.strip(',') + '\n'

    for corpus in corpora:
        line = f'{abv_text[corpus]},'
        for n_topic in n_topics:
            for type_ in types:
                line += f'{results[corpus][type_][str(n_topic)]["ll_per_token_norm"]:.4f},'
        output += line.strip(',') + '\n'
        
    with open('corpus/ll.csv', 'w') as f:
        f.write(output.strip())
    df = pd.read_csv('corpus/ll.csv')
    df.fillna('', inplace=True) # replace NaN
    display(df)
        
    # output silhouette
    output = ''
    line = f'Silhouette,' # heads
    for n_topic in n_topics:
        for type_ in types:
            line += f'{abv_text[type_]}-{n_topic},'
    output += line.strip(',') + '\n'

    for corpus in corpora:
        line = f'{abv_text[corpus]},'
        for n_topic in n_topics:
            for type_ in types:
                line += f'{results[corpus][type_][str(n_topic)][str(n_key)]["silhouette"]},'
        output += line.strip(',') + '\n'
        
    with open('corpus/sil.csv', 'w') as f:
        f.write(output.strip())
    df = pd.read_csv('corpus/sil.csv')
    df.fillna('', inplace=True) # replace NaN
    display(df)

    # output merged percentage
    output = ''
    line = f'Merged Percentage,' # heads
    for type_ in types[1:]:
            line += f'{abv_text[type_]},'
    output += line.strip(',') + '\n'

    for corpus in corpora:
        line = f'{abv_text[corpus]},'
        for type_ in types[1:]:
            line += f'{100*results[corpus][type_]["merged"]:.2f},'
        output += line.strip(',') + '\n'
    
    with open('corpus/merged.csv', 'w') as f:
        f.write(output.strip())
    df = pd.read_csv('corpus/merged.csv')
    display(df)