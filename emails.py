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
import math
from collections import defaultdict
from multiprocessing import Pool
from pprint import PrettyPrinter
from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer

import pybloom_live
import marisa_trie

from graph import Node

def preprocess_emails(tokenizer='split'):
    """Preprocess ENRON emails,
    saving tokenized emails with their hashes in a shelf,
    and ngrams with their count in a trie.
    Must be run before anything else."""

    shelf_filename = 'emails_{}.shelve'.format(tokenizer)
    try:
        os.remove(shelf_filename)
    except FileNotFoundError:
        pass
    
    paths = (f[0]+'/'+g for f in os.walk('maildir/') for g in f[2])
    paths = (path for path in paths if os.path.isfile(path))
    paths, paths_ = itertools.tee(paths)
    # paths = itertools.islice(paths, 100)

    print('processing emails...')
    with Pool() as pool:
        email_pipeline = pool.imap_unordered(email_to_str_list, paths)
        email_pipeline = (thread for thread in email_pipeline if thread is not None)
        email_pipeline = (msg for thread in email_pipeline for msg in thread)
        email_pipeline = pool.imap_unordered(functools.partial(tokenize, tokenizer=tokenizer), email_pipeline)
        email_count = 0
        unique_email_count = 0
        with shelve.open(shelf_filename) as emails:
            for i, (tokens_hash, tokens) in enumerate(email_pipeline):
                if tokens_hash not in emails:
                    bloom_filter = pybloom_live.BloomFilter(capacity=len(set(tokens)), error_rate=0.1)
                    for token in tokens:
                        bloom_filter.add(token)
                    emails[tokens_hash] = (tokens, bloom_filter)
                    unique_email_count += 1
                if i%1000 == 0:
                    print('processing email:', i)
                email_count += 1

    print('saved emails')
    print('total emails: {}'.format(len(list(paths_))))
    print('total parsed emails: {}'.format(email_count))
    print('unique parsed emails: {}'.format(unique_email_count))

def build_trie(ngram_length=3, tokenizer='split'):
    trie_filename = 'marisa_{}.trie'.format(tokenizer)
    try:
        os.remove(trie_filename)
    except FileNotFoundError:
        pass

    ngram_counter = defaultdict(lambda: 0)
    i = 0
    with shelve.open('emails_{}.shelve'.format(tokenizer)) as emails:
        for h in emails:
            i += 1
            tokens, _ = emails[h]
            for ngram, count in count_ngrams(tokens, ngram_length).items():
                ngram_counter[ngram] += count
            if i%1000 == 0:
                    print('processing email:', i)
                    
    print('unique n-grams: {}'.format(len(ngram_counter)))
    print('building trie...')
    trie = marisa_trie.RecordTrie(
        'I',
        ((k, (v,)) for k, v in ngram_counter.items()), order=marisa_trie.WEIGHT_ORDER)
    print('saving trie...')
    trie.save(trie_filename)
    print('saved trie')

def md5_hash(msg):
    """Return the hex string representation of a md5 hash of a string"""
    return hashlib.md5(msg.encode('utf-8')).hexdigest()

def email_to_str_list(path):
    """Return the emails in an email thread file as strings"""
    with open(path) as thread_file:
        try:
            thread = email.message_from_file(thread_file)
            emails = (msg for msg in thread.walk() if not msg.is_multipart())
            return [msg.get_payload() for msg in emails]
        except UnicodeDecodeError:
            print('cannot parse:', path)

def tokenize(msg, tokenizer):
    """Return the tokens from a string"""
    if tokenizer == 'simple':
        tokens = msg.split(' ')
    elif tokenizer == 'split':
        tokens = msg.split()
    elif tokenizer == 'moses':
        tokens = MosesDetokenizer().unescape_xml(MosesTokenizer().tokenize(msg, return_str=True)).split(' ')
    return (md5_hash(detokenize(tokens, tokenizer)), tokens)

def detokenize(tokens, tokenizer):
    """Return the tokens from a string"""
    if tokenizer == 'simple' or tokenizer == 'split':
        msg = ' '.join(tokens)
    elif tokenizer == 'moses':
        msg = MosesDetokenizer().detokenize(tokens, return_str=True)
    return msg

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

def find_ngrams(partial_ngram, bloom_filter, ngram_length):
    """Return ngrams from a trie that match a partially forgotten ngram"""
    #split ngram into known-(partially) unkown part
    try:
        split = partial_ngram.index(None)
    except ValueError as e:
        return [partial_ngram]
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

def make_emails(tokens, bloom_filter, ngram_length):
    graph_levels = [[] for i in range(len(tokens))]

    possible_ngrams = find_ngrams(tokens[:ngram_length], bloom_filter, ngram_length)
    for ngram in possible_ngrams:
        old_node = None
        for j, gram in enumerate(ngram):
            if old_node is not None:
                new_node = old_node.add_child(gram)
            else:
                new_node = Node(gram)
            graph_levels[j].append(new_node)
            old_node = new_node

    node_indices = [0 for i in range(len(tokens))]
    level = ngram_length
    while node_indices[0] < len(graph_levels[0]):
        # if we have gone through all parents, increase grand-parent index and go back one level
        if node_indices[level-1] == len(graph_levels[level-1]):
            node_indices[level-2] += 1
            level -= 1
            continue
        token = tokens[level]
        parent_node = graph_levels[level-1][node_indices[level-1]]
        # sometimes we filter out ngrams or don't generate any, so we need to skip "empty" parents
        if len(parent_node.parents) > 0:
            prefix = parent_node.generate_first_ngram(ngram_length-1)
            # generate n-grams
            if token is None:
                possible_ngrams = find_ngrams(prefix+[token], bloom_filter, ngram_length)
                for ngram in possible_ngrams:
                    new_node = parent_node.add_child(ngram[-1])
                    graph_levels[level].append(new_node)
            # forward filtering
            else:
                if ' '.join(prefix+[token]) in trie:
                    new_node = parent_node.add_child(token)
                    graph_levels[level].append(new_node)
        # if we have reached the end of the tokens, generate emails,
        # increase parent index, and go back one level
        if level == (len(tokens) - 1):
            node_indices[level-1] += 1
            for node in parent_node.children:
                yield from node.generate_ngrams(len(tokens))
        else:
            level += 1

def forget_email(tokens, ratio):
    """Return a tokenized string with some randomly forgotten tokens"""
    email_length = len(tokens)
    # if we don't want to forget the first token:
    # forget = random.sample(range(1, email_length), min(1, int(ratio*email_length)))
    forget = random.sample(range(email_length), max(1, int(ratio*email_length)))
    return [token if i not in forget else None for i, token in enumerate(tokens)]

def recall_email(tokens, md5, ngram_length, bloom_filter, tokenizer):
    """Return the number of emails generated from a partially forgotten email"""
    msgs = make_emails(tokens, bloom_filter, ngram_length)
    found = False
    count = 0
    for msg in msgs:
        count += 1
        if md5_hash(detokenize(msg, tokenizer)) == md5:
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
             'tokenizer: {}, '
             'sample_size: {}, '
             'max_email_len: {}, '
             'email_bins: {}, '
             'use_bloom_filter: {}, '
             'ngram_length: {}, '
             'compare_bloom_filter: {}').format(
                 run_config['time'],
                 run_config['tokenizer'],
                 run_config['sample_size'],
                 run_config['max_email_len'],
                 run_config['email_bins'],
                 run_config['use_bloom_filter'],
                 run_config['ngram_length'],
                 run_config['compare_bloom_filter']),
        ])

def email_stats(item, ratio, run_config):
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

    try:
        if run_config['use_bloom_filter']:
            result['bloom'] = recall_email(
                tokens, md5, run_config['ngram_length'], bloom_filter, tokenizer=run_config['tokenizer'])
            result['bloom_time'] = (datetime.datetime.now()-time_start).total_seconds()

            if run_config['compare_bloom_filter']:
                result['no_bloom'] = recall_email(
                    tokens, md5, run_config['ngram_length'], bloom_filter=None, tokenizer=run_config['tokenizer'])
                result['no_bloom_time'] = (datetime.datetime.now()-time_start).total_seconds()
        else:
            result['no_bloom'] = recall_email(
                tokens, md5, run_config['ngram_length'], bloom_filter=None, tokenizer=run_config['tokenizer'])
            result['no_bloom_time'] = (datetime.datetime.now()-time_start).total_seconds()
    except:
        print('Exception while processing:')
        print(result)
        raise

    return result

def get_binned_email(run_config, emails):
    min_email_len = math.ceil(1/run_config['ratio_step'])
    bin_step = (run_config['max_email_len']-min_email_len)/run_config['email_bins']
    email_bin_bounds = [int(round(bin_step*i+min_email_len)) for i in range(run_config['email_bins']+1)]
    cumulative_bin_sizes = [int(round(run_config['sample_size']/run_config['email_bins']*i)) for i in range(run_config['email_bins']+1)]
    email_bin_sizes = [cumulative_bin_sizes[i]-cumulative_bin_sizes[i-1] for i in range(1,len(cumulative_bin_sizes))]
    original_items = []
    for i in range(run_config['email_bins']):
        new_items = list(itertools.islice(
            ((h, m[0], m[1]) for h, m in emails.items() if
            (email_bin_bounds[i] <= len(m[0]) <= email_bin_bounds[i+1])),
            email_bin_sizes[i]))
        original_items += new_items
    return original_items

trie = None
def main(run_config, clear_csv, verbose):
    """Write data about recalling emails to a csv file"""
    global trie
    random.seed(0) #for reproduceability
    run_config['time'] = datetime.datetime.now().isoformat()
    steps = round(run_config['max_ratio']/run_config['ratio_step'])
    ratios = tuple((step+1)*run_config['ratio_step'] for step in range(steps))

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
    trie.load('marisa_{}.trie'.format(run_config['tokenizer']))

    print('getting email sample')
    items = None
    with shelve.open('emails_{}.shelve'.format(run_config['tokenizer'])) as emails:
        items = get_binned_email(run_config, emails)

    print('processing emails')
    for ratio in ratios:
        with Pool() as pool:
            i = 0
            results = pool.imap_unordered(
                functools.partial(email_stats, run_config=run_config, ratio=ratio),
                items,
                chunksize=5)
            for result in results:
                i += 1
                print('processed: {:.2f}%'.format(100*i/run_config['sample_size']))
                if verbose:
                    PrettyPrinter().pprint(result)
                print_to_csv(result, run_config)

DEFAULT_RUN_CONFIG = {
    'tokenizer':'simple',
    'sample_size':100,
    'ratio_step':0.1,
    'max_ratio':0.9,
    'max_email_len':200,
    'email_bins': 10,
    'ngram_length':3,
    'use_bloom_filter':True,
    'compare_bloom_filter':True,
}

if __name__ == '__main__':
    main(DEFAULT_RUN_CONFIG, clear_csv=True, verbose=False)
