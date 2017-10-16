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
import uuid
import json

from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, Process, Manager
from pprint import PrettyPrinter
from nltk.tokenize.moses import MosesTokenizer, MosesDetokenizer

import pybloom_live
import marisa_trie

def get_paths(run_data):
    paths = {}

    if 'tokenizer' in run_data:
        paths['emails'] = 'emails/tokenizer_{}.shelve'.format(run_data['tokenizer'])
        if 'ngram_length' in run_data:
            option_str = 'tokenizer_{}_ngram-length_{}'.format(
                run_data['tokenizer'], run_data['ngram_length'])
            paths['trie'] = 'tries/{}.marisa_trie'.format(option_str)
            paths['reverse_trie'] = 'tries/{}_reversed.marisa_trie'.format(option_str)
    if 'sample_id' in run_data:
        if run_data['sample_id'] == '':
            sample_id = str(uuid.uuid4())
        else:
            sample_id = run_data['sample_id']
        paths['sample_id'] = sample_id
        paths['original_sample'] = 'samples/{}_original'.format(sample_id)
        if 'forget_method' in run_data:
            paths['forgotten_sample'] = 'samples/{}_forget-method_{}'.format(
                sample_id, run_data['forget_method'])
    if 'use_bloom_filter' in run_data and run_data['use_bloom_filter']:
        paths['bloom_filters'] = 'bloom_filters/error_rate_{:.5f}.shelve'.format(
            run_data['bloom_error_rate'])
    if 'start_time' in run_data:
        paths['results'] = 'results/{}'.format(run_data['start_time'])

    for path_type, path in paths.items():
        if path_type != 'sample_id':
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.mkdir(dirname)

    return paths

def save_emails(run_data):
    print('saving emails')

    paths = get_paths(run_data)
    if os.path.exists(paths['emails']):
        os.remove(paths['emails'])
    
    print('\tgetting files')
    email_paths = (
        os.path.join(dirpath, filename) 
        for dirpath, dirnames, filenames in os.walk('maildir/') 
        for filename in filenames)
    email_paths = filter(os.path.isfile, email_paths)
    email_paths, total_emails = itertools.tee(email_paths)
    total_emails = len(list(total_emails))
    # email_paths = itertools.islice(email_paths, 100)

    print('\tparsing emails')
    with Pool() as pool:
        email_pipeline = pool.imap_unordered(email_to_str_list, email_paths)
        email_pipeline = (thread for thread in email_pipeline if thread is not None)
        email_pipeline = (msg for thread in email_pipeline for msg in thread)
        email_pipeline = pool.imap_unordered(
            functools.partial(tokenize, tokenizer=run_data['tokenizer']), email_pipeline)
        parsed_count = 0
        unique_count = 0
        with shelve.open(paths['emails']) as emails:
            for i, (md5, tokens) in enumerate(email_pipeline):
                parsed_count += 1
                if md5 not in emails:
                    emails[md5] = tokens
                    unique_count += 1
                if i%1000 == 0:
                    print('\t\tparsed {} emails'.format(i))

    print('finished saving emails')
    print('total emails: {}'.format(total_emails))
    print('parsed emails: {}'.format(parsed_count))
    print('unique parsed emails: {}'.format(unique_count))

def save_bloom_filters(run_data):
    paths = get_paths(run_data)
    if os.path.exists(paths['bloom_filters']):
        os.remove(paths['bloom_filters'])

    print('calculating bloom filters...')
    with shelve.open(paths['bloom_filters']) as bloom_filters:
        with shelve.open(paths['emails']) as emails:
            for i, (md5, tokens) in enumerate(emails.items()):
                bloom_filter = pybloom_live.BloomFilter(
                    capacity=len(set(tokens)), error_rate=run_data['bloom_error_rate'])
                for token in tokens:
                    bloom_filter.add(token)
                bloom_filters[md5] = bloom_filter
                if i%1000 == 0:
                    print('\tcalculated {} bloom filters'.format(i))

    print('finished calculating emails')

def save_tries(run_data):
    print('building tries')
    paths = get_paths(run_data)

    for reverse in (False, True):
        if reverse:
            print('\tbuilding reverse trie')
            path = paths['reverse_trie']
        else:
            print('\tbuilding forwards trie')
            path = paths['trie']
        if os.path.exists(path):
            os.remove(path)

        ngram_counter = defaultdict(lambda: 0)
        with shelve.open(paths['emails']) as emails:
            for i, (md5, tokens) in enumerate(emails.items()):
                for ngram, count in count_ngrams(tokens, run_data['ngram_length']).items():
                    if reverse:
                        ngram = ngram.split(' ')
                        ngram.reverse()
                        ngram = ' '.join(ngram)
                    ngram_counter[ngram] += count
                if i%1000 == 0:
                    print('\t\tprocessed {} emails'.format(i))

        trie = marisa_trie.RecordTrie(
            'I',
            ((k, (v,)) for k, v in ngram_counter.items()), order=marisa_trie.WEIGHT_ORDER)
        trie.save(path)
        print('\tfinished building trie')
        print('unique n-grams: {}'.format(len(ngram_counter)))

def save_original_sample(run_data):
    print('sampling original emails')
    paths = get_paths(run_data)
    bins = [{} for bin in run_data['bin_bounds']]

    with shelve.open(paths['emails']) as emails:
        for (md5, tokens) in emails.items():
            for i, bin_bound in enumerate(run_data['bin_bounds']):
                if bin_bound[0] <= len(tokens) < bin_bound[1]:
                    if len(bins[i]) < run_data['bin_size']:
                        bins[i][md5] = tokens
                    break
            if all(len(bin_) == run_data['bin_size'] for bin_ in bins):
                break

    with shelve.open(paths['original_sample']) as sample:
        for (md5, tokens) in bins[i].items():
            sample[md5] = tokens
    print('original emails sampled')

def save_forgotten_sample(run_data):
    print('sampling forgotten emails')
    paths = get_paths(run_data)

    with shelve.open(paths['forgotten_sample']) as forgotten_sample:
        with shelve.open(paths['original_sample']) as original_sample:
            for i, ratio in enumerate(run_data['ratios']):
                sample_by_ratio = {}
                for (md5, tokens) in original_sample.items():
                    forgotten_email, frequency_threshold = forget_email(tokens, ratio, run_data)
                    item = {
                        'md5': md5,
                        'original_email': tokens,
                        'length': len(tokens),
                        'ratio': ratio,
                        'forgotten_email': forgotten_email,
                        'frequency_threshold': frequency_threshold,
                        'bloom_filter': None,
                    }
                    sample_by_ratio[md5] = item
                forgotten_sample[str(ratio)] = sample_by_ratio

    print('forgotten emails sampled')

def md5_hash(msg):
    return hashlib.md5(msg.encode('utf-8')).hexdigest()

def email_to_str_list(path):
    with open(path) as thread_file:
        try:
            thread = email.message_from_file(thread_file)
            emails = (msg for msg in thread.walk() if not msg.is_multipart())
            return [msg.get_payload() for msg in emails]
        except UnicodeDecodeError:
            pass
            # print('cannot parse:', path)

def tokenize(msg, tokenizer):
    if tokenizer == 'simple':
        tokens = msg.split(' ')
    elif tokenizer == 'split':
        tokens = msg.split()
    elif tokenizer == 'moses':
        tokens = MosesDetokenizer().unescape_xml(MosesTokenizer().tokenize(msg, return_str=True)).split(' ')
    return (md5_hash(detokenize(tokens, tokenizer)), tokens)

def detokenize(tokens, tokenizer):
    if tokenizer == 'simple' or tokenizer == 'split':
        msg = ' '.join(tokens)
    elif tokenizer == 'moses':
        msg = MosesDetokenizer().detokenize(tokens, return_str=True)
    return msg

def count_ngrams(tokens, ngram_length):
    counter = defaultdict(lambda: 0)
    for ngram in tokens_to_ngrams(tokens, ngram_length):
        counter[' '.join(ngram)] += 1
    return counter

def tokens_to_ngrams(tokens, ngram_length):
    for i in range(len(tokens)):
        ngram = tokens[i:i+ngram_length]
        for j in range(len(ngram)):
            yield ngram[:j+1]

def find_ngrams(partial_ngram, run_data, bloom_filter=None, frequency_threshold=None):
    # split ngram into known-(partially) unkown part
    forgotten_indices = [i for i, gram in enumerate(partial_ngram) if gram is None]
    remembered_indices = [i for i in range(run_data['ngram_length']) if i not in forgotten_indices]
    if len(forgotten_indices) == 0:
        raise Exception(partial_ngram, 'not supposed to look up ngram without forgotten tokens')
    else:
        prefix = partial_ngram[:forgotten_indices[0]]
    # get ngrams from trie
    if len(prefix) == 0:
        ngrams = run_data['trie'].iteritems()
    else:
        ngrams = run_data['trie'].iteritems(' '.join(prefix)+' ')
    # filter ngrams by length
    ngrams = ((gram.split(' '), count) for gram, (count,) in ngrams)
    ngrams = ((gram, _) for gram, _ in ngrams if len(gram) == run_data['ngram_length'])
    # filter by remembered tokens
    ngrams = (
        (gram, _) for gram, _ in ngrams if 
        all(gram[i] == partial_ngram[i] for i in remembered_indices))
    # filter ngrams with bloom filter
    if bloom_filter is not None:
        ngrams = (
            (gram, _) for gram, _ in ngrams if 
            all(gram[i] in bloom_filter for i in forgotten_indices))
    #filter by frequency threshold
    if frequency_threshold is not None:
        ngrams = (
            (gram, frequency) for gram, frequency in ngrams if 
            all(run_data['trie'][gram[i]] >= frequency_threshold for i in forgotten_indices))
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
    if not run_data['use_bloom_filter'] and run_data['bloom_filter'] is not None:
        raise Exception('use_bloom_filter is False but bloom_filter is not None')
    if not run_data['use_frequency_threshold'] and run_data['frequency_threshold'] is not None:
        raise Exception('use_frequency_threshold is False but frequency_threshold is not None')
        
    tokens = item['forgotten_email']
    ngram_length = run_data['ngram_length']
    graph_levels = [[] for i in range(len(tokens))]

    prefix = tokens[:ngram_length]
    # if the first token is forgotten, use reverse trie to enable prefix search
    if tokens[0] is None:
        prefix.reverse()
        possible_ngrams = find_ngrams(
            prefix, run_data, item['bloom_filter'], item['frequency_threshold'])
    elif None in tokens[:ngram_length]:
        possible_ngrams = find_ngrams(
            prefix, run_data, item['bloom_filter'], item['frequency_threshold'])
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
                    target_ngram, run_data, item['bloom_filter'], item['frequency_threshold'])
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

def forget_email(tokens, ratio, run_data):
    number_to_forget = max(1, round(ratio*len(tokens)))
    frequency_threshold = None
    if run_data['forget_method'] == 'random':
        forget = random.sample(range(len(tokens)), number_to_forget)
    elif run_data['forget_method'] == 'frequency':
        frequencies = [(i, run_data['trie'][token]) for i, token in enumerate(tokens)]
        frequencies = sorted(frequencies, key=lambda item: item[1], reverse=True)
        forget = [i for (i, token) in frequencies[:number_to_forget]]
        frequency_threshold = frequencies[number_to_forget-1][1]
    forgotten_email = [token if i not in forget else None for i, token in enumerate(tokens)]
    return forgotten_email, frequency_threshold

def recall_email(item, run_data):
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

def print_to_csv(item, run_data, path):
    try:
        with open(path, 'a') as csv_file:
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
                run_data['ngram_length'],
                run_data['bloom_error_rate'],
                run_data['use_bloom_filter'],
                run_data['use_hash'],
                run_data['max_emails_generated'],
                run_data['ratios'],
                run_data['bin_bounds'],
                run_data['bin_size'],
                run_data['sample_id'],
                run_data['forget_method'],
                run_data['use_frequency_threshold'],
            ])
    except Exception as e:
        print(item)
        raise e

def main(run_data):
    run_data['start_time'] = datetime.datetime.now().isoformat()

    paths = get_paths(run_data)
    run_data['sample_id'] = paths['sample_id']

    if not os.path.exists(paths['emails']):
        save_emails(run_data)
    if not os.path.exists(paths['trie']) or not os.path.exists(paths['reverse_trie']):
        save_tries(run_data)
    run_data['trie'] = marisa_trie.RecordTrie('I')
    run_data['trie'].load(paths['trie'])
    run_data['reverse_trie'] = marisa_trie.RecordTrie('I')
    run_data['reverse_trie'].load(paths['reverse_trie'])
    if run_data['use_bloom_filter'] and not os.path.exists(paths['bloom_filters']):
        save_bloom_filters(run_data)
    if not os.path.exists(paths['original_sample']):
        save_original_sample(run_data)
    if not os.path.exists(paths['forgotten_sample']):
        save_forgotten_sample(run_data)

    print('run_data:')
    PrettyPrinter().pprint(run_data)
    print()

    with open(paths['results'], 'w') as csv_file:
        csv.writer(csv_file).writerow([
            'item md5',
            'item ratio',
            'item count',
            'item runtime',
            'item original_email',
            'item forgotten_email',
            'item length',
            'run start_time',
            'run tokenizer',
            'run ngram_length',
            'run bloom_error_rate',
            'run use_bloom_filter',
            'run use_hash',
            'run max_emails_generated',
            'run ratios',
            'run bin_bounds',
            'run bin_size',
            'run sample_id',
            'run forget_method',
            'run use_frequency_threshold',
        ])

    sample_size = run_data['bin_size'] * len(run_data['bin_bounds'])
    # maxtasksperchild prevents memory leak from growing too much
    with Pool(maxtasksperchild=1) as pool:
        with shelve.open(paths['forgotten_sample']) as sample:
            for ratio in run_data['ratios']:
                i = 0
                print('processing ratio: {}'.format(ratio))
                items = sample[str(ratio)].values()
                if run_data['use_bloom_filter']:
                    with shelve.open(paths['bloom_filters']) as bloom_filters:
                        for item in items:
                            item['bloom_filter'] = bloom_filters[item['md5']]
                if not run_data['use_frequency_threshold']:
                    for item in items:
                        item['frequency_threshold'] = None
                target = functools.partial(recall_email, run_data=run_data)
                processed_items = pool.imap_unordered(target, items)
                for item in processed_items:
                    i += 1
                    print('\tprocessed: {:.2f}%'.format(100*i/sample_size))
                    print_to_csv(item, run_data, paths['results'])
                    # PrettyPrinter().pprint(item)

DEFAULT_RUN_DATA = {
    'tokenizer': 'moses',
    'ngram_length': 3,
    'bloom_error_rate': 0.01,
    'max_emails_generated': 50000,
    'bin_size': 100,
    'ratios': [round(0.1 * i, 1) for i in range(1, 6)],
    'bin_bounds': [[i,i+10] for i in range(10, 100, 10)],
    'use_bloom_filter': True,
    'use_hash': False,
    'sample_id': '',
    'forget_method': 'random',
    'use_frequency_threshold': False,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_data', nargs='?', type=json.loads)
    parser.set_defaults(run_data=DEFAULT_RUN_DATA)
    run_data = vars(parser.parse_args())['run_data']
    for param, default in DEFAULT_RUN_DATA.items():
        if param not in run_data:
            run_data[param] = default
    for param in run_data:
        if param not in DEFAULT_RUN_DATA:
            raise Exception('This run_data option does not exist')
    if run_data['use_frequency_threshold'] and run_data['forget_method'] != 'frequency':
        raise Exception('Cannot use frequency threshold if forget_method is not set to frequency')
    main(args['run_data'])
