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

import pybloomfilter
import marisa_trie

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

    print('processing emails...')
    with Pool() as pool:
        email_pipeline = pool.imap_unordered(email_to_str_list, paths)
        email_pipeline = (
            email for thread in email_pipeline for email in thread if email is not None)
        email_pipeline = pool.imap_unordered(tokenize_email, email_pipeline)
        with shelve.open('emails.shelve') as emails:
            for i, (tokens_hash, tokens) in enumerate(email_pipeline):
                if tokens_hash not in emails:
                    bloom_filter_path = 'tmp/bloom_'+tokens_hash
                    bloom_filter = pybloomfilter.BloomFilter(len(tokens), 0.1, bloom_filter_path)
                    bloom_filter.update(tokens)
                    bloom_filter.sync()
                    with open(bloom_filter_path, 'rb') as bloom_filter_file:
                        emails[tokens_hash] = (tokens, bloom_filter_file.read())
                    os.remove(bloom_filter_path)
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
            emails = (email for email in thread.walk() if not email.is_multipart())
            return [email.get_payload() for email in emails]
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

def make_emails(tokens, trie, bloom_filter, ngram_length, length=None):
    """Recursively generate possible emails from a partially forgotten email"""
    if length is None:
        length = len(tokens)
    #ngram has not forgotten parts
    if None not in tokens:
        yield tokens
    #email length is equal to ngram length
    elif len(tokens) == ngram_length:
        for ngram in find_ngrams(tokens, trie, bloom_filter, ngram_length):
            yield ngram
    #only last token is forgotten
    elif None not in tokens[:-1] and tokens[-1] is None:
        for ngram in make_emails(tokens[-ngram_length:], trie, bloom_filter, ngram_length, length):
            yield tokens[:-ngram_length]+ngram
    #forgotten token(s) are not at the end
    else:
        for msg in make_emails(tokens[:-1], trie, bloom_filter, ngram_length, length):
            msg.append(tokens[-1])
            for msg in make_emails(msg, trie, bloom_filter, ngram_length, length):
                yield msg

def forget_email(tokens, ratio):
    """Return a tokenized string with some reandomly forgotten tokens"""
    email_length = len(tokens)
    forget = random.sample(range(email_length), int((1-ratio)*email_length))
    return [token if i in forget else None for i, token in enumerate(tokens)]

def recall_email(tokens, trie, md5, ngram_length, bloom_filter):
    """Return the number of generated emails from a partially forgotten email that were hashed
    before finding the correct one"""
    hashed_count = 0
    for msg in make_emails(tokens, trie, bloom_filter, ngram_length):
        hashed_count += 1
        if md5_hash(msg) == md5:
            return hashed_count
    #email should always be recalled
    raise Exception('Could not reconstruct email.')

def email_stats(item, ratio, trie, run_config):
    """Return a dictionary with information about a recalled email"""
    result = {}
    md5, msg, bloom_filter_bin = item
    if run_config['use_bloom_filter']:
        bloom_filter_path = 'tmp/bloom_'+md5
        with open(bloom_filter_path, 'wb') as bloom_filter_file:
            bloom_filter_file.write(bloom_filter_bin)
        bloom_filter = pybloomfilter.BloomFilter.open(bloom_filter_path)
    result['md5'] = md5
    result['length'] = len(msg)
    result['ratio'] = ratio

    tokens = forget_email(msg, ratio)
    if run_config['use_bloom_filter']:
        result['bloom_hashed'] = recall_email(
            tokens, trie, md5, run_config['ngram_length'], bloom_filter)
        if run_config['compare_bloom_filter']:
            result['no_bloom_hashed'] = recall_email(
                tokens, trie, md5, run_config['ngram_length'], bloom_filter=None)
    else:
        result['no_bloom_hashed'] = recall_email(
            tokens, trie, md5, run_config['ngram_length'], bloom_filter=None)
    if run_config['use_bloom_filter']:
        os.remove(bloom_filter_path)
    return result

def main(run_config, verbose=False, clear=False):
    """Write data about recalling emails to a csv file"""
    random.seed(0) #for reproduceability
    time = datetime.datetime.now().isoformat()
    steps = round(run_config['max_ratio']/run_config['ratio_step'])
    ratios = tuple((step+1)*run_config['ratio_step'] for step in range(steps))
    try:
        os.mkdir('tmp')
    except FileExistsError:
        pass
    if not os.path.isfile('stats.csv') or clear:
        with open('stats.csv', 'w') as csv_file:
            csv.writer(csv_file).writerow([
                'md5',
                'number of tokens',
                'ratio forgotten',
                'emails hashed - bloom',
                'emails hashed - no bloom',
                'run info',
            ])
    print('opening trie...')
    trie = marisa_trie.RecordTrie('I')
    trie.load('marisa.trie')

    print('gathering stats...')
    with shelve.open('emails.shelve') as emails:
        for ratio in ratios:
            print('processing ratio {:.5}...'.format(ratio))
            with Pool() as pool:
                items = (
                    (h, i[0], i[1]) for h, i in emails.items() if
                    run_config['ngram_length'] <= len(i[0]) <= run_config['max_email_len'])
                results = pool.imap_unordered(
                    functools.partial(
                        email_stats,
                        run_config=run_config,
                        ratio=ratio,
                        trie=trie),
                    items)
                for result in itertools.islice(results, run_config['sample_size']):
                    if verbose:
                        PrettyPrinter().pprint(result)
                    with open('stats.csv', 'a') as csv_file:
                        csv.writer(csv_file).writerow([
                            result['md5'],
                            result['length'],
                            '{:.5}'.format(result['ratio']),
                            result.get('bloom_hashed', ''),
                            result.get('no_bloom_hashed', ''),
                            ('time: {}, '
                             'sample_size: {}, '
                             'max_email_len: {}, '
                             'use_bloom_filter: {}, '
                             'ngram_length: {}, '
                             'compare_bloom_filter: {}').format(
                                 time,
                                 run_config['sample_size'],
                                 run_config['max_email_len'],
                                 run_config['use_bloom_filter'],
                                 run_config['ngram_length'],
                                 run_config['compare_bloom_filter']),
                        ])
    shutil.rmtree('tmp')

if __name__ == '__main__':
    RUN_CONFIG = {
        'sample_size':100,
        'ratio_step':0.1,
        'max_ratio':0.9,
        'max_len':30,
        'ngram_length':3,
        'use_bloom_filter':True,
        'compare_bloom_filter':True,}
    main(RUN_CONFIG)
