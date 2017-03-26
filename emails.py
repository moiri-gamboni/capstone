"""Text Forgit implementation on the ENRON email dataset
"""
import os
import email
import random
import csv
import hashlib
import shelve
import itertools
import functools
import re
import shutil
import datetime
from collections import defaultdict
from multiprocessing import Pool
from pprint import PrettyPrinter

import pybloom_live
import marisa_trie

from graph import Node

def preprocess(ngram_length=3):
    """Preprocess ENRON emails,
    saving tokenized emails with their hashes in a shelf,
    and ngrams with their count in a trie.
    Must be run before anything else."""
    try:
        os.mkdir('tmp')
    except FileExistsError:
        pass
    try:
        os.remove('emails.shelve')
    except FileNotFoundError:
        pass
    try:
        os.remove('marisa.trie')
    except FileNotFoundError:
        pass

    ngram_counter = defaultdict(lambda: 0)
    paths = (f[0]+'/'+g for f in os.walk('maildir/') for g in f[2])
    paths = (path for path in paths if os.path.isfile(path))
    # paths = itertools.islice(paths, 100)

    print('processing emails...')
    with Pool() as pool:
        email_pipeline = pool.imap_unordered(email_to_str_list, paths)
        email_pipeline = (thread for thread in email_pipeline if thread is not None)
        email_pipeline = (msg for thread in email_pipeline for msg in thread)
        email_pipeline = pool.imap_unordered(tokenize_email, email_pipeline)
        with shelve.open('emails.shelve') as emails:
            for i, (tokens_hash, tokens) in enumerate(email_pipeline):
                if tokens_hash not in emails:
                    bloom_filter = pybloom_live.BloomFilter(capacity=len(set(tokens)), error_rate=0.1)
                    for token in tokens:
                        bloom_filter.add(token)
                    emails[tokens_hash] = (tokens, bloom_filter)
                for ngram, count in count_ngrams(tokens, ngram_length).items():
                    ngram_counter[ngram] += count
                if i%1000 == 0:
                    print('processing email:', i)
    print('saved emails')

    print('building trie...')
    trie = marisa_trie.RecordTrie(
        'I',
        ((k, (v,)) for k, v in ngram_counter.items()), order=marisa_trie.WEIGHT_ORDER)
    print('saving trie...')
    trie.save('marisa.trie')
    print('saved trie')
    shutil.rmtree('tmp')

def md5_hash(tokens):
    """Return the hex string representation of a md5 hash of a list of tokens"""
    return hashlib.md5(' '.join(tokens).encode('utf-8')).hexdigest()

def email_to_str_list(path):
    """Return the emails in an email thread file as strings"""
    with open(path) as thread_file:
        try:
            thread = email.message_from_file(thread_file)
            emails = (msg for msg in thread.walk() if not msg.is_multipart())
            return [msg.get_payload() for msg in emails]
        except UnicodeDecodeError:
            print('cannot parse:', path)

def tokenize_email(msg):
    """Return the tokens from a string"""
    tokens = re.split(r'\s+', msg)
    tokens = [token for token in tokens if token != '']
    return (md5_hash(tokens), tokens)

def count_ngrams(tokens, ngram_length):
    """Return a dictonary with ngrams as keys and their occurences as values"""
    counter = defaultdict(lambda: 0)
    for ngram in tokens_to_ngrams(tokens, ngram_length):
        counter[' '.join(ngram)] += 1
    return counter

def tokens_to_ngrams(tokens, ngram_length):
    """Return all possiple ngrams of length ngram_length or less of a tokenized string"""
    for i in range(len(tokens)):
        ngram = tokens[i:i+ngram_length]
        for j in range(len(ngram)):
            yield ngram[:j+1]

def find_ngrams(partial_ngram, trie, bloom_filter, ngram_length):
    """Return ngrams from a trie that match a partially forgotten ngram"""
    #split ngram into known-(partially) unkown part
    split = partial_ngram.index(None)
    prefix = partial_ngram[:split]
    suffix = partial_ngram[split:ngram_length]
    #get ngrams from trie
    if len(prefix) > 0:
        ngrams = trie.iteritems(' '.join(prefix)+' ')
    else:
        ngrams = trie.iteritems()
    #filter ngrams by length and suffix match
    ngrams = (
        item for item in ngrams if
        len(item[0].split(' ')) == ngram_length and
        all(item[0].split(' ')[split:][i] == c for i, c in enumerate(suffix) if c is not None))
    #filter ngrams with bloom filter
    if bloom_filter is not None:
        ngrams = (
            item for item in ngrams if
            all(word in bloom_filter for word in item[0].split(' ')))
    #sort ngrams in descending order of occurence
    ngrams = sorted(ngrams, key=lambda item: item[1][0], reverse=True)
    return [ngram.split(' ') for ngram, v in ngrams]

def join_graph_level(node_list, ngram_length):
    unjoined = set(node_list)
    while len(unjoined) > 0:
        to_join = unjoined.pop()
        to_remove = set()
        for node in unjoined:
            if to_join.join(node, ngram_length):
                to_remove.add(node)
        unjoined -= to_remove

def make_emails_breadth(tokens, trie, bloom_filter, ngram_length, join=False):
    graph_levels = [[] for i in range(len(tokens))]
    if None in tokens[:ngram_length-1]:
        possible_ngrams = find_ngrams(tokens[:ngram_length], trie, bloom_filter, ngram_length)
        for ngram in possible_ngrams:
            old_node = None
            for j, gram in enumerate(ngram):
                if old_node is not None:
                    new_node = old_node.add_child(gram)
                else:
                    new_node = Node(gram)
                graph_levels[j].append(new_node)
                old_node = new_node
        start = ngram_length
    else:
        old_node = None
        for j in range(ngram_length-1):
            if old_node is not None:
                new_node = old_node.add_child(tokens[j])
            else:
                new_node = Node(tokens[j])
            graph_levels[j].append(new_node)
            old_node = new_node
        start = ngram_length - 1

    # x = 0
    for i in range(start, len(tokens)):
        token = tokens[i]
        for node in graph_levels[i-1]:
            if len(node.parents) > 0:
                # x += 1
                prefix = node.generate_first_ngram(ngram_length-1)
                if token is None:
                    possible_ngrams = find_ngrams(prefix+[token], trie, bloom_filter, ngram_length)
                    for ngram in possible_ngrams:
                        new_node = node.add_child(ngram[-1])
                        graph_levels[i].append(new_node)
                else:
                    if ' '.join(prefix+[token]) in trie:
                        new_node = node.add_child(token)
                        graph_levels[i].append(new_node)
        if join:
            join_graph_level(graph_levels[i], ngram_length)

    msgs = [msg for node in graph_levels[-1] for msg in node.generate_ngrams(len(tokens))]
    # print('number of emails:', len(msgs))
    # print('x', x)
    return msgs

def make_emails_depth(tokens, trie, bloom_filter, ngram_length, join=False):
    graph_levels = [[] for i in range(len(tokens))]

    if None in tokens[:ngram_length-1]:
        possible_ngrams = find_ngrams(tokens[:ngram_length], trie, bloom_filter, ngram_length)
        for ngram in possible_ngrams:
            old_node = None
            for j, gram in enumerate(ngram):
                if old_node is not None:
                    new_node = old_node.add_child(gram)
                else:
                    new_node = Node(gram)
                graph_levels[j].append(new_node)
                old_node = new_node
        start = ngram_length
    else:
        old_node = None
        for j in range(ngram_length-1):
            if old_node is not None:
                new_node = old_node.add_child(tokens[j])
            else:
                new_node = Node(tokens[j])
            graph_levels[j].append(new_node)
            old_node = new_node
        start = ngram_length - 1

    node_indices = [0 for i in range(len(tokens))]
    level = start
    while node_indices[0] < len(graph_levels[0]):
        if node_indices[level-1] == len(graph_levels[level-1]):
            node_indices[level-2] += 1
            level -= 1
            continue
        token = tokens[level]
        parent_node = graph_levels[level-1][node_indices[level-1]]
        if len(parent_node.parents) > 0:
            prefix = parent_node.generate_first_ngram(ngram_length-1)
            if token is None:
                possible_ngrams = find_ngrams(prefix+[token], trie, bloom_filter, ngram_length)
                for ngram in possible_ngrams:
                    new_node = parent_node.add_child(ngram[-1])
                    graph_levels[level].append(new_node)
            else:
                if ' '.join(prefix+[token]) in trie:
                    new_node = parent_node.add_child(token)
                    graph_levels[level].append(new_node)
        if join:
            join_graph_level(graph_levels[level], ngram_length)
        if level == (len(tokens) - 1):
            node_indices[level-1] += 1
            for node in parent_node.children:
                for msg in node.generate_ngrams(len(tokens)):
                    yield msg
        else:
            level += 1

def forget_email(tokens, ratio):
    """Return a tokenized string with some reandomly forgotten tokens"""
    email_length = len(tokens)
    forget = random.sample(range(1, email_length), min(1, int(ratio*email_length)))
    # with first token possibly forgotten:
    # forget = random.sample(range(email_length), min(1, int(ratio*email_length)))
    return [token if i not in forget else None for i, token in enumerate(tokens)]

def recall_email(tokens, trie, md5, ngram_length, bloom_filter, verbose=False):
    """Return the number of emails generated from a partially forgotten email"""
    msgs = make_emails_depth(tokens, trie, bloom_filter, ngram_length)
    found = False
    count = 0
    for msg in msgs:
        if verbose:
            print(msg)
        count += 1
        if md5_hash(msg) == md5:
            found = True
            if bloom_filter is None:
                return count
    #email should always be recalled
    if found:
        return count
    raise Exception('Could not reconstruct email.')

def print_to_csv(result, run_config):
    with open('stats.csv', 'a') as csv_file:
        csv.writer(csv_file).writerow([
            result['md5'],
            result['length'],
            '{:.5}'.format(result['ratio']),
            result.get('bloom', ''),
            result.get('bloom_time', ''),
            result.get('no_bloom', ''),
            result.get('no_bloom_time', ''),
            result['original_email'],
            result['forgotten_email'],
            ('time: {}, '
             'sample_size: {}, '
             'max_email_len: {}, '
             'use_bloom_filter: {}, '
             'ngram_length: {}, '
             'compare_bloom_filter: {}').format(
                 run_config['time'],
                 run_config['sample_size'],
                 run_config['max_email_len'],
                 run_config['use_bloom_filter'],
                 run_config['ngram_length'],
                 run_config['compare_bloom_filter']),
        ])

def email_stats(item, ratio, trie, run_config):
    """Return a dictionary with information about a recalled email"""
    time_start = datetime.datetime.now()
    result = {}
    md5, msg, bloom_filter = item
    tokens = forget_email(msg, ratio)

    result['md5'] = md5
    result['length'] = len(msg)
    result['ratio'] = ratio
    result['original_email'] = msg
    result['forgotten_email'] = tokens

    print(md5)
    print(msg)
    print(tokens)
    try:
        if run_config['use_bloom_filter']:
            result['bloom'] = recall_email(
                tokens, trie, md5, run_config['ngram_length'], bloom_filter)
            result['bloom_time'] = (datetime.datetime.now()-time_start).total_seconds()

            if run_config['compare_bloom_filter']:
                result['no_bloom'] = recall_email(
                    tokens, trie, md5, run_config['ngram_length'], bloom_filter=None)
                result['no_bloom_time'] = (datetime.datetime.now()-time_start).total_seconds()
        else:
            result['no_bloom'] = recall_email(
                tokens, trie, md5, run_config['ngram_length'], bloom_filter=None)
            result['no_bloom_time'] = (datetime.datetime.now()-time_start).total_seconds()
    except:
        print('Exception while processing:')
        print(result)
        raise

    print('*'*80)
    return result

def main(run_config, clear_csv, verbose):
    """Write data about recalling emails to a csv file"""
    random.seed(0) #for reproduceability
    run_config['time'] = datetime.datetime.now().isoformat()
    steps = round(run_config['max_ratio']/run_config['ratio_step'])
    ratios = tuple((step+1)*run_config['ratio_step'] for step in range(steps))
    try:
        os.mkdir('tmp')
    except FileExistsError:
        pass
    if not os.path.isfile('stats.csv') or clear_csv:
        with open('stats.csv', 'w') as csv_file:
            csv.writer(csv_file).writerow([
                'md5',
                'number of tokens',
                'ratio forgotten',
                'emails generated - bloom',
                'runtime - bloom',
                'emails generated - no bloom',
                'runtime - no bloom',
                'original email',
                'forgotten email',
                'run info',
            ])
    trie = marisa_trie.RecordTrie('I')
    trie.load('marisa.trie')

    with Pool() as pool:
        with shelve.open('emails.shelve') as emails:
            results = [None for i in range(len(ratios))]
            original_items = itertools.islice(
                ((h, i[0], i[1]) for h, i in emails.items() if
                (run_config['ngram_length'] <= len(i[0]) <= run_config['max_email_len']) and 
                (run_config['ratio_step'] * len(i[0]) >= 1)), 
                run_config['sample_size'])        
            for i, ratio in enumerate(ratios):
                items, original_items = itertools.tee(original_items)
                results[i] = pool.imap_unordered(
                    functools.partial(email_stats, run_config=run_config, ratio=ratio, trie=trie),
                    items,
                    chunksize=1+int(run_config['sample_size']/os.cpu_count()/len(ratios)))
            for results_by_ratio in results:
                for result in results_by_ratio:
                    if verbose:
                        PrettyPrinter().pprint(result)
                    print_to_csv(result, run_config)

    shutil.rmtree('tmp')

DEFAULT_RUN_CONFIG = {
    'sample_size':500,
    'ratio_step':0.1,
    'max_ratio':0.9,
    'max_email_len':200,
    'ngram_length':3,
    'use_bloom_filter':True,
    'compare_bloom_filter':True,}

if __name__ == '__main__':
    main(DEFAULT_RUN_CONFIG, clear_csv=True, verbose=False)
