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
import datetime
import math
import argparse
import copy
import json

from collections import defaultdict
from multiprocessing import Pool, Process, Manager
from pprint import PrettyPrinter
from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer

import pybloom_live
import marisa_trie

def get_paths(run_data):
    filepath = {}
    try:
        filepath['shelf'] = 'emails_{}_{:.5f}.shelve'.format(
            run_data['tokenizer'], run_data['bloom_error_rate'])
    except KeyError:
        pass
    try:
        filepath['reverse_trie'] = 'reverse_trie_{}_{}'.format(
            run_data['tokenizer'], run_data['ngram_length'])
        filepath['trie'] = 'trie_{}_{}'.format(
            run_data['tokenizer'], run_data['ngram_length'])
    except KeyError:
        pass
    sample_data = (
        'tokenizer',
        'sample_size',
        'ratio_step',
        'max_ratio',
        'max_email_len',
        'total_email_bins',
        'ngram_length',)
    try:
        filepath['sample'] = ('sample' + '_{}'*len(sample_data) + '.shelve').format(
            *(run_data[d] for d in sample_data))
    except KeyError:
        pass
    return filepath

def preprocess_emails(tokenizer, bloom_error_rate):
    """Preprocess ENRON emails,
    saving tokenized emails with their hashes in a shelf,
    and ngrams with their count in a trie.
    Must be run before anything else."""

    shelf_filename = get_paths(
        {'tokenizer':tokenizer, 'bloom_error_rate':bloom_error_rate})['shelf']
    if os.path.isfile(shelf_filename):
        os.remove(shelf_filename)
    
    paths = (f[0]+'/'+g for f in os.walk('maildir/') for g in f[2])
    paths = (path for path in paths if os.path.isfile(path))
    paths, paths_ = itertools.tee(paths)
    # paths = itertools.islice(paths, 100)

    print('processing emails...')
    with Pool() as pool:
        email_pipeline = pool.imap_unordered(email_to_str_list, paths)
        email_pipeline = (thread for thread in email_pipeline if thread is not None)
        email_pipeline = (msg for thread in email_pipeline for msg in thread)
        email_pipeline = pool.imap_unordered(
            functools.partial(tokenize, tokenizer=tokenizer), email_pipeline)
        email_count = 0
        unique_email_count = 0
        with shelve.open(shelf_filename) as emails:
            for i, (tokens_hash, tokens) in enumerate(email_pipeline):
                if tokens_hash not in emails:
                    bloom_filter = pybloom_live.BloomFilter(
                        capacity=len(set(tokens)), error_rate=bloom_error_rate)
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

def build_trie(ngram_length, tokenizer):
    paths = get_paths(
        {'tokenizer':tokenizer, 'ngram_length':ngram_length})
    for reverse in (True, False):
        if reverse:
            trie_filename = paths['reverse_trie']
        else:
            trie_filename = paths['trie']

        if os.path.isfile(trie_filename):
            os.remove(trie_filename)

        ngram_counter = defaultdict(lambda: 0)
        i = 0
        with shelve.open('emails_{}.shelve'.format(tokenizer)) as emails:
            for h in emails:
                i += 1
                tokens, _ = emails[h]
                for ngram, count in count_ngrams(tokens, ngram_length).items():
                    if reverse:
                        ngram = ngram.split(' ')
                        ngram.reverse()
                        ngram = ' '.join(ngram)
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

def find_ngrams(partial_ngram, bloom_filter, ngram_length, trie):
    """Return ngrams from a trie that match a partially forgotten ngram"""
    # split ngram into known-(partially) unkown part
    forgotten_indices = [i for i, gram in enumerate(partial_ngram) if gram is None]
    remembered_indices = [i for i in range(ngram_length) if i not in forgotten_indices]
    if len(forgotten_indices) == 0:
        raise Exception(partial_ngram, 'not supposed to look up ngram without forgotten tokens')
    else:
        prefix = partial_ngram[:forgotten_indices[0]]
    # get ngrams from trie
    if len(prefix) == 0:
        ngrams = trie.iteritems()
    else:
        ngrams = trie.iteritems(' '.join(prefix)+' ')
    # filter ngrams by length
    ngrams = ((gram.split(' '), count) for gram, (count,) in ngrams)
    ngrams = ((gram, _) for gram, _ in ngrams if len(gram) == ngram_length)
    # filter by remembered tokens
    ngrams = (
        (gram, _) for gram, _ in ngrams if 
        all(gram[i] == partial_ngram[i] for i in remembered_indices))
    # filter ngrams with bloom filter
    if bloom_filter is not None:
        ngrams = (
            (gram, _) for gram, _ in ngrams if 
            all(gram[i] in bloom_filter for i in forgotten_indices))
    # sort ngrams in ascending order of occurence
    ngrams = sorted(ngrams, key=lambda item: item[1], reverse=True)
    return [gram for gram, _ in ngrams]

def get_node_ngram(node, ngram_length):
    tmp = node
    ngram = []
    while len(ngram) < ngram_length and tmp is not None:
        ngram.insert(0, tmp[0])
        tmp = tmp[1]
    return ngram

def split_independent_parts(tokens, ngram_length):
    forgotten_indices = [i for i, token in enumerate(tokens) if token is None]
    tmp_ranges = [(i-(ngram_length-1), i+(ngram_length-1)) for i in forgotten_indices]
    combined_ranges = []
    current_range = tmp_ranges[0]
    for r in tmp_ranges:
        if current_range[1] - r[0] >= ngram_length - 1:
            current_range = (current_range[0], r[1])
        else:
            combined_ranges.append(current_range)
            current_range = r
    combined_ranges.append(current_range)
    if combined_ranges[0][0] < 0:
        combined_ranges[0] = (0, combined_ranges[0][1])
    if combined_ranges[-1][1] >= len(tokens):
        combined_ranges[-1] = (combined_ranges[-1][0], len(tokens)-1)
    split_emails = [tokens[r[0]:r[1]+1] for r in combined_ranges]
    return split_emails

def count_emails(item, run_data):
    count = 1
    split_emails = split_independent_parts(item['forgotten_email'], run_data['ngram_length'])
    for msg in split_emails:
        tmp_count = 0
        new_item = item.copy()
        new_item['forgotten_email'] = msg
        new_item['length'] = len(msg)
        new_item['original_email'] = None
        new_item['md5'] = None
        for msg in make_emails(new_item, run_data):
            tmp_count += 1
        count *= tmp_count
        if count >= run_data['max_emails_generated']:
            return -1
    return count

def make_emails(item, run_data):
    tokens = item['forgotten_email']
    if run_data['use_bloom_filter']:
        bloom_filter = item['bloom_filter']
    else:
        bloom_filter = None
    ngram_length = run_data['ngram_length']

    graph_levels = [[] for i in range(len(tokens))]

    prefix = tokens[:ngram_length]
    # if the first token is forgotten, use reverse trie to enable prefix search
    if tokens[0] is None:
        prefix.reverse()
        possible_ngrams = find_ngrams(prefix, bloom_filter, ngram_length, run_data['reverse_trie'])
    elif None in tokens[:ngram_length]:
        possible_ngrams = find_ngrams(prefix, bloom_filter, ngram_length, run_data['trie'])
    else:
        possible_ngrams = [tokens[:ngram_length]]

    # add found ngrams to 'graph'
    for ngram in possible_ngrams:
        # if we used the reverse trie, we need to re-reverse the ngram
        if tokens[0] is None:
            ngram.reverse()
        old_node = None
        for j, gram in enumerate(ngram):
            new_node = (gram, old_node)
            if j == ngram_length-1:
                graph_levels[j].append(new_node)
            old_node = new_node

    if len(tokens) > ngram_length:
        level = ngram_length
        while any(len(graph_levels[i]) > 0 for i in range(ngram_length)):
            # if we have gone through all parents, pop grand-parent and go back one level
            if len(graph_levels[level-1]) == 0:
                graph_levels[level-2].pop()
                level -= 1
                continue
            token = tokens[level]
            parent_node = graph_levels[level-1][-1]
            target_ngram = get_node_ngram(parent_node, ngram_length-1)+[token]
            # generate n-grams
            if token is None:
                possible_ngrams = find_ngrams(
                    target_ngram, bloom_filter, ngram_length, run_data['trie'])
                for ngram in possible_ngrams:
                    new_node = (ngram[-1], parent_node)
                    graph_levels[level].append(new_node)
            # forward filtering
            else:
                if ' '.join(target_ngram) in run_data['trie']:
                    new_node = (token, parent_node)
                    graph_levels[level].append(new_node)
                else:
                    graph_levels[level-1].pop()
                    continue
            # if we have reached the end of the tokens, generate emails,
            # pop parent, and go back one level
            if level == (len(tokens) - 1):
                while len(graph_levels[-1]) > 0:
                    node = graph_levels[-1].pop()
                    msg = get_node_ngram(node, len(tokens))
                    for i, t in enumerate(msg):
                        if tokens[i] is not None and t != tokens[i]:
                            raise Exception(tokens, msg, 'Generated email does not match original')
                    yield msg
                graph_levels[-2].pop()
            else:
                level += 1
    else:
        while len(graph_levels[-1]) > 0:
            node = graph_levels[-1].pop()
            msg = get_node_ngram(node, len(tokens))
            for i, t in enumerate(msg):
                if tokens[i] is not None and t != tokens[i]:
                    raise Exception(tokens, msg, 'Generated email does not match original')
            yield msg

def forget_email(tokens, ratio):
    """Return a tokenized string with some randomly forgotten tokens"""
    email_length = len(tokens)
    forget = random.sample(range(email_length), max(1, int(ratio/100*email_length)))
    return [token if i not in forget else None for i, token in enumerate(tokens)]

def recall_email(item, run_data):
    """Return the number of emails generated from a partially forgotten email"""
    new_item = item.copy()
    time_start = datetime.datetime.now()
    if run_data['use_hash']:
        msgs = make_emails(item, run_data)
        count = 0
        md5s = set()
        for msg in msgs:
            count += 1
            if count == run_data['max_emails_generated']:
                new_item['runtime'] = -1
                new_item['count'] = -1
                return new_item
            test_md5 = md5_hash(detokenize(msg, run_data['tokenizer']))
            if test_md5 in md5s:
                raise Exception((item, 'Duplicate email generated'))
            else:
                md5s.add(test_md5)
            if item['md5'] == test_md5:
                runtime = (datetime.datetime.now()-time_start).total_seconds()
                new_item['runtime'] = runtime
                new_item['count'] = count
                return new_item
        raise Exception((item, 'No email matched hash'))
    else:
        count = count_emails(item, run_data)
        runtime = (datetime.datetime.now()-time_start).total_seconds()
        new_item['count'] = count
        if count == -1:
            new_item['runtime'] = -1
        else:
            new_item['runtime'] = runtime
        return new_item

def print_to_csv(item, run_data):
    try:
        with open('stats_{}.csv'.format(run_data['start_time']), 'a') as csv_file:
            csv.writer(csv_file).writerow([
                item['md5'],
                item['ratio'],
                item['count'],
                item['runtime'],
                item['original_email'],
                item['forgotten_email'],
                item['length'],
                run_data['start_time'],
                run_data['tokenizer'],
                run_data['sample_size'],
                run_data['ratio_step'],
                run_data['max_ratio'],
                run_data['max_email_len'],
                run_data['total_email_bins'],
                run_data['ngram_length'],
                run_data['bloom_error_rate'],
                run_data['use_bloom_filter'],
                run_data['use_hash'],
                run_data['max_emails_generated'],
            ])
    except Exception as e:
        print(item)
        print(e)

def get_binned_emails(emails, run_data):
    min_email_len = math.ceil(100/run_data['ratio_step'])
    bin_step = (run_data['max_email_len']-min_email_len)/run_data['total_email_bins']
    email_bin_bounds = [
        int(round(bin_step*i+min_email_len)) 
        for i in range(run_data['total_email_bins']+1)]
    cumulative_bin_sizes = [
        int(round(run_data['sample_size']/run_data['total_email_bins']*i)) 
        for i in range(run_data['total_email_bins']+1)]
    email_bin_sizes = [
        cumulative_bin_sizes[i]-cumulative_bin_sizes[i-1] 
        for i in range(1,len(cumulative_bin_sizes))]
    original_items = []
    for i in range(run_data['total_email_bins']):
        new_items = (e for e in emails.items() if 
            email_bin_bounds[i] <= len(e[1][0]) < email_bin_bounds[i+1])
        new_items = list(itertools.islice((
            {
                'md5': h,
                'original_email': m[0], 
                'bloom_filter': m[1],
                'length': len(m[0])
            } for h, m in new_items),
            email_bin_sizes[i]))
        original_items += new_items
    return original_items

def get_email_sample(run_data, ratios):
    print('getting email sample')
    paths = get_paths(run_data)
    with shelve.open(paths['shelf']) as emails:
        items = get_binned_emails(emails, run_data)
    with shelve.open(paths['sample']) as sample:
        for ratio in ratios:
            sample_by_ratio = {}
            for item in items:
                item['ratio'] = ratio
                item['forgotten_email'] = forget_email(item['original_email'], ratio)
                sample_by_ratio[item['md5']] = item
            sample[str(ratio)] = sample_by_ratio

def main(run_data):
    """Write data about recalling emails to a csv file"""
    run_data['start_time'] = datetime.datetime.now().isoformat()
    steps = round(run_data['max_ratio']/run_data['ratio_step'])
    ratios = tuple((step+1)*run_data['ratio_step'] for step in range(steps))
    with open('stats_{}.csv'.format(run_data['start_time']), 'w') as csv_file:
        csv.writer(csv_file).writerow([
            'md5',
            'ratio',
            'count',
            'runtime',
            'original_email',
            'forgotten_email',
            'length',
            'run start_time'
            'run tokenizer',
            'run sample_size',
            'run ratio_step',
            'run max_ratio',
            'run max_email_len',
            'run total_email_bins',
            'run ngram_length',
            'run bloom_error_rate',
            'run use_bloom_filter',
            'run use_hash',
            'run max_emails_generated',
        ])
    paths = get_paths(run_data)
    if not os.path.isfile(paths['shelf']):
        preprocess_emails(run_data['tokenizer'], run_data['bloom_error_rate'])
    if not os.path.isfile(paths['trie']) or not os.path.isfile(paths['reverse_trie']):
        build_trie(run_data['ngram_length'], run_data['tokenizer'])
    if not os.path.isfile(paths['sample']):
        get_email_sample(run_data, ratios)

    run_data['trie'] = marisa_trie.RecordTrie('I')
    run_data['trie'].load(paths['trie'])
    run_data['reverse_trie'] = marisa_trie.RecordTrie('I')
    run_data['reverse_trie'].load(paths['reverse_trie'])

    with shelve.open(paths['sample']) as sample:
        # maxtasksperchild prevents memory leak from growing too much
        with Pool(maxtasksperchild=1) as pool:
            for ratio in ratios:
                i = 0
                print('processing ratio: {}'.format(ratio))
                items = sample[str(ratio)].values()
                target = functools.partial(recall_email, run_data=run_data)
                processed_items = pool.imap_unordered(target, items)
                for item in processed_items:
                    i += 1
                    print('\tprocessed: {:.2f}%'.format(100*i/run_data['sample_size']))
                    print_to_csv(item, run_data)
                    # PrettyPrinter().pprint(item)

DEFAULT_RUN_DATA = {
    'tokenizer': 'moses',
    'sample_size': 100,
    'ratio_step': 10,
    'max_ratio': 90,
    'max_email_len': 100,
    'total_email_bins': 9,
    'ngram_length': 3,
    'bloom_error_rate': 0.01,
    'use_bloom_filter': True,
    'use_hash': False,
    'max_emails_generated': 100000,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for param in DEFAULT_RUN_DATA:
        default = DEFAULT_RUN_DATA[param]
        parser.add_argument('--{}'.format(param), default=default, type=type(default))
    args = vars(parser.parse_args())
    run_data = {}
    for param in DEFAULT_RUN_DATA:
        run_data[param] = args[param]
    main(run_data)
