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
        email_pipeline = pool.imap_unordered(
            functools.partial(tokenize, tokenizer=tokenizer), email_pipeline)
        email_count = 0
        unique_email_count = 0
        with shelve.open(shelf_filename) as emails:
            for i, (tokens_hash, tokens) in enumerate(email_pipeline):
                if tokens_hash not in emails:
                    bloom_filter = pybloom_live.BloomFilter(
                        capacity=len(set(tokens)), error_rate=0.1)
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

def build_trie(ngram_length=3, tokenizer='split', reverse=False):
    if reverse:
        trie_filename = 'marisa_{}.reverse_trie'.format(tokenizer)
    else:
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
        tokens = MosesDetokenizer().unescape_xml(
            MosesTokenizer().tokenize(msg, return_str=True)).split(' ')
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
    #split ngram into known-(partially) unkown part
    forgotten_indices = [i for i, gram in enumerate(partial_ngram) if gram is None]
    remembered_indices = [i for i in range(ngram_length) if i not in forgotten_indices]
    if len(forgotten_indices) == 0:
        raise Exception(partial_ngram, 'not supposed to look up ngram without forgotten tokens')
    else:
        prefix = partial_ngram[:forgotten_indices[0]]
        if len(prefix) == 0:
            raise Exception(partial_ngram, 'not supposed to look up ngram without prefix')
    #get ngrams from trie
    ngrams = trie.iteritems(' '.join(prefix)+' ')
    #filter ngrams by length
    ngrams = ((gram.split(' '), count) for gram, (count,) in ngrams)
    ngrams = ((gram, _) for gram, _ in ngrams if len(gram) == ngram_length)
    #filter by remembered tokens
    ngrams = (
        (gram, _) for gram, _ in ngrams if 
        all(gram[i] == partial_ngram[i] for i in remembered_indices))
    #filter ngrams with bloom filter
    if bloom_filter is not None:
        ngrams = (
            (gram, _) for gram, _ in ngrams if 
            all(gram[i] in bloom_filter for i in forgotten_indices))
    #sort ngrams in ascending order of occurence
    ngrams = sorted(ngrams, key=lambda item: item[1], reverse=True)
    return [gram for gram, _ in ngrams]

def join_graph_level(node_list, ngram_length):
    unjoined = set(node_list)
    while len(unjoined) > 0:
        to_join = unjoined.pop()
        to_remove = set()
        for node in unjoined:
            if to_join.join(node, ngram_length):
                to_remove.add(node)
        unjoined -= to_remove

def get_node_ngram(node, ngram_length):
    tmp = node
    ngram = []
    while len(ngram) < ngram_length and tmp is not None:
        ngram.insert(0, tmp[0])
        tmp = tmp[1]
    return ngram

def make_emails(tokens, bloom_filter, ngram_length, trie, reverse_trie):
    graph_levels = [[] for i in range(len(tokens))]

    prefix = tokens[:ngram_length]
    if tokens[0] is None:
        prefix.reverse()
        possible_ngrams = find_ngrams(prefix, bloom_filter, ngram_length, reverse_trie)
    elif None in tokens[:ngram_length]:
        possible_ngrams = find_ngrams(prefix, bloom_filter, ngram_length, trie)
    else:
        possible_ngrams = [tokens[:ngram_length]]

    for ngram in possible_ngrams:
        if tokens[0] is None:
            ngram.reverse()
        old_node = None
        for j, gram in enumerate(ngram):
            new_node = (gram, old_node)
            if j == ngram_length-1:
                graph_levels[j].append(new_node)
            old_node = new_node

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
            possible_ngrams = find_ngrams(target_ngram, bloom_filter, ngram_length, trie)
            for ngram in possible_ngrams:
                new_node = (ngram[-1], parent_node)
                graph_levels[level].append(new_node)
        # forward filtering
        else:
            if ' '.join(target_ngram) in trie:
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

def forget_email(tokens, ratio):
    """Return a tokenized string with some randomly forgotten tokens"""
    email_length = len(tokens)
    forget = random.sample(range(email_length), max(1, int(ratio*email_length)))
    return [token if i not in forget else None for i, token in enumerate(tokens)]

def recall_email(tokens, md5, ngram_length, bloom_filter, trie, reverse_trie, tokenizer):
    """Return the number of emails generated from a partially forgotten email"""
    msgs = make_emails(tokens, bloom_filter, ngram_length, trie, reverse_trie)
    found = False
    count = 0
    md5s = set()
    with open('tmp/{}'.format(md5), 'a') as f:
        for msg in msgs:
            count += 1
            json.dump(msg, f)
            f.write('\n')
            test_md5 = md5_hash(detokenize(msg, tokenizer))
            if test_md5 in md5s:
                raise Exception((md5, msg, 'Duplicate email generated'))
            else:
                md5s.add(test_md5)
            if test_md5 == md5:
                found = True
                if bloom_filter is None:
                    return count
    if found:
        return count
    #email should always be recalled
    raise Exception(md5, 'Could not reconstruct email.')

def setup_email(item, ratio):
    result = {}
    md5, msg, bloom_filter = item
    tokens = forget_email(msg, ratio)
    result['md5'] = md5
    result['length'] = len(msg)
    result['ratio'] = ratio
    result['original_email'] = msg
    result['forgotten_email'] = tokens
    result['bloom_filter'] = bloom_filter
    return result

def email_stats(result, use_bloom_filter, run_config, trie, reverse_trie):
    """Return a dictionary with information about a recalled email"""
    time_start = datetime.datetime.now()
    with open('tmp/{}'.format(result['md5']), 'w') as f:
        write_result = result.copy()
        del write_result['bloom_filter']
        json.dump(write_result, f)
        f.write('\n')
    count = recall_email(
        tokens=result['forgotten_email'],
        md5=result['md5'],
        bloom_filter=result['bloom_filter'] if use_bloom_filter else None,
        ngram_length=run_config['ngram_length'],
        trie=trie,
        reverse_trie=reverse_trie,
        tokenizer=run_config['tokenizer'])
    time = (datetime.datetime.now()-time_start).total_seconds()
    key = 'bloom' if use_bloom_filter else 'hash'
    result['{}_count'.format(key)] = count
    result['{}_time'.format(key)] = time
    return result

def print_to_csv(result, run_config):
    with open('stats.csv', 'a') as csv_file:
        csv.writer(csv_file).writerow([
            result['md5'],
            result['length'],
            '{:.5}'.format(result['ratio']),
            result.get('bloom_count', ''),
            result.get('bloom_time', ''),
            result.get('hash_count', ''),
            result.get('hash_time', ''),
            result['original_email'],
            result['forgotten_email'],
            ('time: {}, '
             'tokenizer: {}, '
             'sample_size: {}, '
             'max_email_len: {}, '
             'email_bins: {}, '
             'use_bloom_filter: {}, '
             'ngram_length: {}, '
             'use_hash: {}').format(
                 run_config['time'],
                 run_config['tokenizer'],
                 run_config['sample_size'],
                 run_config['max_email_len'],
                 run_config['email_bins'],
                 run_config['use_bloom_filter'],
                 run_config['ngram_length'],
                 run_config['use_hash']),
        ])

def get_binned_email(run_config, emails):
    min_email_len = math.ceil(1/run_config['ratio_step'])
    bin_step = (run_config['max_email_len']-min_email_len)/run_config['email_bins']
    email_bin_bounds = [
        int(round(bin_step*i+min_email_len)) 
        for i in range(run_config['email_bins']+1)]
    cumulative_bin_sizes = [
        int(round(run_config['sample_size']/run_config['email_bins']*i)) 
        for i in range(run_config['email_bins']+1)]
    email_bin_sizes = [
        cumulative_bin_sizes[i]-cumulative_bin_sizes[i-1] 
        for i in range(1,len(cumulative_bin_sizes))]
    original_items = []
    for i in range(run_config['email_bins']):
        new_items = list(itertools.islice(
            ((h, m[0], m[1]) for h, m in emails.items() if
            (email_bin_bounds[i] <= len(m[0]) <= email_bin_bounds[i+1])),
            email_bin_sizes[i]))
        original_items += new_items
    return original_items

def main(run_config, clear_csv, verbose):
    """Write data about recalling emails to a csv file"""
    random.seed(0) #for reproduceability
    run_config['time'] = datetime.datetime.now().isoformat()
    steps = round(run_config['max_ratio']/run_config['ratio_step'])
    ratios = tuple((step+1)*run_config['ratio_step'] for step in range(steps))
    if not os.path.isdir('tmp/'):
        os.mkdir('tmp/')
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
    reverse_trie = marisa_trie.RecordTrie('I')
    reverse_trie.load('marisa_{}.reverse_trie'.format(run_config['tokenizer']))

    print('getting email sample')
    items = None
    with shelve.open('emails_{}.shelve'.format(run_config['tokenizer'])) as emails:
        items = get_binned_email(run_config, emails)

    with Pool(maxtasksperchild=1) as pool:
        for ratio in ratios:
            print('processing ratio: {}'.format(ratio))
            i = 0
            forgotten_emails = list(map(functools.partial(setup_email, ratio=ratio), items))
            target = functools.partial(
                email_stats,
                run_config=run_config,
                trie=trie,
                reverse_trie=reverse_trie)
            results_by_type = []
            if run_config['use_bloom_filter']:
                target = functools.partial(target, use_bloom_filter=True)
                results_by_type.append(pool.imap_unordered(target, forgotten_emails, chunksize=5))
            if run_config['use_hash']:
                target = functools.partial(target, use_bloom_filter=False)
                results_by_type.append(pool.imap_unordered(target, forgotten_emails, chunksize=5))
            for results in results_by_type:
                for result in results:
                    i += 1
                    print('\tprocessed: {:.2f}%'.format(
                        (100*i)/(len(results_by_type)*run_config['sample_size'])))
                    print_to_csv(result, run_config)
                    if verbose:
                        PrettyPrinter().pprint(result)



    # with Manager() as manager:
    #     for ratio in ratios:
    #         print('processing ratio: {}'.format(ratio))
    #         processes = []
    #         forgotten_emails = list(map(functools.partial(setup_email, ratio=ratio), items))
    #         results = manager.dict()
    #         for m in forgotten_emails:
    #             results[m['md5']] = manager.dict(m)
    #         for result in forgotten_emails:
    #             key = result['md5']
    #             target = functools.partial(
    #                 email_stats, 
    #                 result=result,
    #                 run_config=run_config,
    #                 trie=trie,
    #                 managed_dict=results)
    #             if run_config['use_bloom_filter']:
    #                 processes.append(Process(
    #                     target=target,
    #                     kwargs={'use_bloom_filter':True}))
    #                 results[key]['bloom_count'] = 'dropped'
    #                 results[key]['bloom_time'] = 'dropped'
    #             if run_config['use_hash']:
    #                 processes.append(Process(
    #                     target=target,
    #                     kwargs={'use_bloom_filter':False}))
    #                 results[key]['hash_count'] = 'dropped'
    #                 results[key]['hash_time'] = 'dropped'
    #         for i in range(os.cpu_count()):
    #             processes[i].start()
    #         for i in range(len(processes)):
    #             if processes[i].join(5) is None and processes[i].exitcode is None:
    #                 processes[i].terminate()
    #                 print('dropped')
    #             next_process = os.cpu_count()+i
    #             if next_process < len(processes):
    #                 processes[next_process].start()
    #             print('\tprocessed: {:.2f}%'.format(
    #                 100*i/len(processes)))
    #         for result in results.values():
    #             print_to_csv(result, run_config)

DEFAULT_RUN_CONFIG = {
    'tokenizer':'moses',
    'sample_size':100,
    'ratio_step':0.1,
    'max_ratio':0.9,
    'max_email_len':200,
    'email_bins': 10,
    'ngram_length':3,
    'use_bloom_filter':True,
    'use_hash':False,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--clear_csv', default=False)
    parser.add_argument('--verbose', default=False)
    for param in DEFAULT_RUN_CONFIG:
        default = DEFAULT_RUN_CONFIG[param]
        parser.add_argument('--{}'.format(param), default=default, type=type(default))
    args = vars(parser.parse_args())
    run_config = {}
    for param in DEFAULT_RUN_CONFIG:
        run_config[param] = args[param]
    main(run_config, clear_csv=args['clear_csv'], verbose=args['verbose'])
