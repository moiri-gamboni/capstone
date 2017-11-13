import os
import email
import random
import csv
import hashlib
import shelve
import itertools
import functools
import datetime
import argparse
import uuid
import json
import time

from collections import Counter
from multiprocessing import Pool
from pprint import PrettyPrinter

import nltk
nltk.download('perluniprops')
nltk.download('nonbreaking_prefixes')
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
        if run_data['sample_id'] is None:
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
        paths['results'] = 'results/{}.csv'.format(run_data['start_time'])

    for path_type, path in paths.items():
        if path_type != 'sample_id':
            dirname = os.path.dirname(path)
            if not os.path.exists(dirname):
                os.mkdir(dirname)

    return paths

def save_emails(run_data):
    print('saving emails')

    paths = get_paths(run_data)
    
    print('\tgetting email files')
    email_paths = (
        os.path.join(dirpath, filename) 
        for dirpath, dirnames, filenames in os.walk('maildir/') 
        for filename in filenames)
    email_paths = list(filter(os.path.isfile, email_paths))
    total_emails = len(email_paths)

    if run_data['multiprocess']:
        pool = Pool()
        map_f = functools.partial(pool.imap_unordered, chunksize=1000)
    else:
        map_f = map

    email_pipeline = map_f(email_to_str_list, email_paths)
    email_pipeline = (thread for thread in email_pipeline if thread is not None)
    email_pipeline = (msg for thread in email_pipeline for msg in thread)
    email_pipeline = map_f(
        functools.partial(tokenize, tokenizer=run_data['tokenizer']),
        email_pipeline)
    parsed_count = 0
    unique_count = 0
    with shelve.open(paths['emails'], 'n') as emails:
        for i, (md5, tokens) in enumerate(email_pipeline):
            parsed_count += 1
            if md5 not in emails:
                emails[md5] = tokens
                unique_count += 1
            if i%1000 == 0:
                print('\tprocessed {} emails'.format(i))

    if run_data['multiprocess']:
        pool.close()
        pool.join()

    print('finished saving emails')
    print('total emails: {}'.format(total_emails))
    print('parsed emails: {}'.format(parsed_count))
    print('unique parsed emails: {}'.format(unique_count))

def initialize_bloom_filter(item, bloom_error_rate):
    md5, tokens = item
    bloom_filter = pybloom_live.BloomFilter(
        capacity=len(set(tokens)),
        error_rate=bloom_error_rate)
    for token in tokens:
        bloom_filter.add(token)
    return (md5, bloom_filter)

def save_bloom_filters(run_data):
    paths = get_paths(run_data)

    print('calculating bloom filters...')

    if run_data['multiprocess']:
        pool = Pool()
        map_f = functools.partial(pool.imap_unordered, chunksize=1000)
    else:
        map_f = map

    with shelve.open(paths['bloom_filters'], 'n') as bf_shelve:
        with shelve.open(paths['emails'], 'r') as emails:
            bloom_filters = map_f(
                functools.partial(
                    initialize_bloom_filter,
                    bloom_error_rate=run_data['bloom_error_rate']),
                emails.items())
            for i, (md5, bloom_filter) in enumerate(bloom_filters):
                bf_shelve[md5] = bloom_filter
                if i%1000 == 0:
                    print('\tprocessed {} emails'.format(i))

    if run_data['multiprocess']:
        pool.close()
        pool.join()

    print('finished calculating bloom filters')

def count_ngrams(tokens, ngram_length):
    counter = Counter()
    reverse_counter = Counter()
    for i in range(len(tokens)):
        super_ngram = tokens[i:i+ngram_length]
        for j in range(len(super_ngram)):
            ngram = super_ngram[:j+1]
            counter[' '.join(ngram)] += 1
            ngram.reverse()
            reverse_counter[' '.join(ngram)] += 1
    return (counter, reverse_counter)

def save_tries(run_data):
    print('building tries')
    paths = get_paths(run_data)

    forward_counter = Counter()
    backward_counter = Counter()

    if run_data['multiprocess']:
        pool = Pool()
        map_f = functools.partial(pool.imap_unordered, chunksize=1000)
    else:
        map_f = map

    with shelve.open(paths['emails']) as emails:
        partial_counters = map_f(
            functools.partial(count_ngrams, ngram_length=run_data['ngram_length']),
            emails.values())
        for i, (forward_partial, backward_partial) in enumerate(partial_counters):
            forward_counter.update(forward_partial)
            backward_counter.update(backward_partial)
            if i%1000 == 0:
                print('\tprocessed {} emails'.format(i))

    if run_data['multiprocess']:
        pool.close()
        pool.join()

    for reverse in (False, True):
        if reverse:
            path = paths['reverse_trie']
            counter = backward_counter
        else:
            path = paths['trie']
            counter = forward_counter

        if os.path.exists(path):
            os.remove(path)

        trie = marisa_trie.RecordTrie(
            'I',
            ((k, (v,)) for k, v in counter.items()),
            order=marisa_trie.WEIGHT_ORDER)
        trie.save(path)
    print('finished building tries')
    print('unique n-grams: {}'.format(len(forward_counter)))

def save_original_sample(run_data):
    print('sampling original emails')
    paths = get_paths(run_data)

    if run_data['use_bins']:
        bins = [{} for bin in run_data['bin_bounds']]
        with shelve.open(paths['emails'], 'r') as emails:
            md5s = list(emails.keys())
            random.shuffle(md5s)
            for md5 in md5s:
                tokens = emails[md5]
                for i, bin_bound in enumerate(run_data['bin_bounds']):
                    if bin_bound[0] <= len(tokens) < bin_bound[1]:
                        if len(bins[i]) < run_data['bin_size']:
                            bins[i][md5] = tokens
                        break
                if all(len(bin_) == run_data['bin_size'] for bin_ in bins):
                    break

        with shelve.open(paths['original_sample'], 'n') as sample:
            for bin in bins:
                for (md5, tokens) in bin.items():
                    sample[md5] = tokens

    else:
        with shelve.open(paths['emails'], 'r') as emails:
            with shelve.open(paths['original_sample'], 'n') as sample:
                md5s = list(emails.keys())
                random.shuffle(md5s)
                for md5 in md5s:
                    tokens = emails[md5]
                    if round(1/run_data['ratios'][0]) <= len(tokens) <= run_data['max_email_length']:
                        sample[md5] = tokens
                        if len(sample) == run_data['sample_size']:
                            break

    with open('samples/metadata', 'a+') as f:
        metadata = get_sample_metadata(run_data)
        json.dump([paths['sample_id'], metadata], f)
        f.write('\n')

    print('original emails sampled with sample_id:')
    print(paths['sample_id'])
    return paths['sample_id']

def get_sample_metadata(run_data):
    return {
        'use_bins': run_data['use_bins'],
        'bin_bounds': run_data['bin_bounds'],
        'tokenizer': run_data['tokenizer'],
        'bin_size': run_data['bin_size'],
        'ratios': run_data['ratios'],
        'max_email_length': run_data['max_email_length'],
        'sample_size': run_data['sample_size'],
    }

def save_forgotten_sample(run_data):
    print('sampling forgotten emails')
    paths = get_paths(run_data)

    if run_data['forget_method'] == 'frequency':
        if 'trie' not in run_data:
            run_data['trie'] = marisa_trie.RecordTrie('I')
            run_data['trie'].load(paths['trie'])

    with shelve.open(paths['forgotten_sample'], 'n') as forgotten_sample:
        with shelve.open(paths['original_sample'], 'r') as original_sample:
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
    return (md5_hash(' '.join(tokens)), tokens)

def find_ngrams(partial_ngram, item, run_data, reverse=False):
    # split ngram into known-(partially) unkown part
    if reverse:
        trie = run_data['reverse_trie']
    else:
        trie = run_data['trie']

    forgotten_indices = [i for i, gram in enumerate(partial_ngram) if gram is None]
    remembered_indices = [i for i in range(run_data['ngram_length']) if i not in forgotten_indices]
    if len(forgotten_indices) == 0:
        raise Exception(partial_ngram, 'not supposed to look up ngram without forgotten tokens')
    else:
        prefix = partial_ngram[:forgotten_indices[0]]
    # get ngrams from trie
    if len(prefix) == 0:
        ngrams = trie.iteritems()
    else:
        ngrams = trie.iteritems(' '.join(prefix) + ' ')
    # filter ngrams by length
    ngrams = ((gram.split(' '), count) for gram, (count,) in ngrams)
    ngrams = ((gram, _) for gram, _ in ngrams if len(gram) == run_data['ngram_length'])
    # filter by remembered tokens
    ngrams = (
        (gram, _) for gram, _ in ngrams if 
        all(gram[i] == partial_ngram[i] for i in remembered_indices))
    # filter ngrams with bloom filter
    if item['bloom_filter'] is not None:
        ngrams = (
            (gram, _) for gram, _ in ngrams if 
            all(gram[i] in item['bloom_filter'] for i in forgotten_indices))
    #filter by frequency threshold
    if item['frequency_threshold'] is not None:
        ngrams = (
            (gram, frequency) for gram, frequency in ngrams if 
            all(run_data['trie'][gram[i]][0][0] >= item['frequency_threshold'] for i in forgotten_indices))
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

def split_by_fixed_length(item, run_data):
    email_parts = []
    lower_bounds = list(range(0, len(item['original_email']), run_data['partial_hash_length']))
    if len(item['original_email']) - lower_bounds[-1] < run_data['ngram_length']:
        lower_bounds[-1] = len(item['original_email']) - run_data['ngram_length']
    for lower_bound in lower_bounds:
        found = False
        original_tokens = item['original_email'][lower_bound:lower_bound + run_data['partial_hash_length']]
        forgotten_tokens = item['forgotten_email'][lower_bound:lower_bound + run_data['partial_hash_length']]
        if None not in forgotten_tokens:
            continue
        email_part = item.copy()
        email_part['original_email'] = original_tokens
        email_part['forgotten_email'] = forgotten_tokens
        email_part['length'] = len(original_tokens)
        email_part['md5'] = md5_hash(' '.join(original_tokens))
        email_parts.append(email_part)
    return email_parts

def split_by_independent_parts(item, run_data):
    email_parts = []
    forgotten_indices = [i for i, token in enumerate(item['forgotten_email']) if token is None]
    tmp_ranges = [(i-(run_data['ngram_length']-1), i+(run_data['ngram_length']-1)) for i in forgotten_indices]
    combined_ranges = []
    current_range = tmp_ranges[0]
    for r in tmp_ranges:
        if current_range[1] - r[0] >= run_data['ngram_length'] - 1:
            current_range = (current_range[0], r[1])
        else:
            combined_ranges.append(current_range)
            current_range = r
    combined_ranges.append(current_range)
    if combined_ranges[0][0] < 0:
        combined_ranges[0] = (0, combined_ranges[0][1])
    if combined_ranges[-1][1] >= len(item['forgotten_email']):
        combined_ranges[-1] = (combined_ranges[-1][0], len(item['forgotten_email'])-1)
    for r in combined_ranges:
        original_tokens = item['original_email'][r[0]:r[1]+1]
        forgotten_tokens = item['forgotten_email'][r[0]:r[1]+1]
        email_part = item.copy()
        email_part['original_email'] = original_tokens
        email_part['forgotten_email'] = forgotten_tokens
        email_part['length'] = len(original_tokens)
        email_part['md5'] = md5_hash(' '.join(original_tokens))
        email_parts.append(email_part)
    return email_parts

def make_emails(item, run_data):
    if not run_data['use_bloom_filter'] and item['bloom_filter'] is not None:
        raise Exception('use_bloom_filter is False but bloom_filter is not None')
    if not run_data['use_frequency_threshold'] and item['frequency_threshold'] is not None:
        raise Exception('use_frequency_threshold is False but frequency_threshold is not None')
        
    tokens = item['forgotten_email']
    ngram_length = run_data['ngram_length']
    graph_levels = [[] for i in range(len(tokens))]

    first_ngram = tokens[:ngram_length]
    # if the first token is forgotten, use reverse trie to enable prefix search
    if tokens[0] is None:
        first_ngram.reverse()
        possible_ngrams = find_ngrams(first_ngram, item, run_data, reverse=True)
    elif None in tokens[:ngram_length]:
        possible_ngrams = find_ngrams(first_ngram, item, run_data)
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
                possible_ngrams = find_ngrams(target_ngram, item, run_data)
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
        frequencies = [(i, run_data['trie'][token][0][0]) for i, token in enumerate(tokens)]
        frequencies = sorted(frequencies, key=lambda item: item[1], reverse=True)
        forget = [i for (i, frequency) in frequencies[:number_to_forget]]
        frequency_threshold = frequencies[number_to_forget-1][1]
    forgotten_email = [token if i not in forget else None for i, token in enumerate(tokens)]
    return forgotten_email, frequency_threshold

def recall_email(item, run_data):
    time_start = time.process_time()
    returned_item = item.copy()
    email_parts = []
    if run_data['use_hash']:
        count = 0
    else:
        count = 1
    if run_data['hash_type'] == 'full':
        email_parts = [item]
    else:
        if run_data['hash_type'] == 'split':
            email_parts = split_by_fixed_length(item, run_data)
        else:
            email_parts = split_by_independent_parts(item, run_data)

    for email_part in email_parts:
        md5s = set()
        tmp_count = 0
        found = False
        for msg in make_emails(email_part, run_data):
            runtime = (time.process_time()-time_start)
            if run_data['use_hash']:
                count += 1
                if runtime > run_data['max_runtime']:
                    returned_item['runtime'] = -1
                    returned_item['emails_generated'] = -1
                    return returned_item
                test_md5 = md5_hash(' '.join(msg))
                if test_md5 in md5s:
                    raise Exception((item, 'Duplicate email generated'))
                else:
                    md5s.add(test_md5)
                if email_part['md5'] == test_md5:
                    found = True
                    break
            else:
                tmp_count += 1

        if run_data['use_hash'] and not found:
            raise Exception((item, 'No email matched hash'))
        if not run_data['use_hash']:
            count *= tmp_count
            if runtime >= run_data['max_runtime']:
                returned_item['runtime'] = -1
                returned_item['emails_generated'] = -1
                return returned_item
 
    runtime = time.process_time()-time_start
    returned_item['runtime'] = runtime
    returned_item['emails_generated'] = count
    return returned_item

def print_to_csv(item, run_data, path):
    with open(path, 'a') as csv_file:
        csv.writer(csv_file).writerow([
            item['md5'],
            item['original_email'],
            item['length'],
            item['ratio'],
            item['forgotten_email'],
            item['frequency_threshold'],
            item['emails_generated'],
            item['runtime'],
            run_data['start_time'],
            run_data['multiprocess'],
            run_data['use_last_sample'],
            run_data['sample_id'],
            run_data['tokenizer'],
            run_data['ngram_length'],
            run_data['max_email_length'],
            run_data['max_runtime'],
            run_data['use_bins'],
            run_data['bin_size'],
            run_data['bin_bounds'],
            run_data['sample_size'],
            run_data['ratios'],
            run_data['forget_method'],
            run_data['use_frequency_threshold'],
            run_data['use_bloom_filter'],
            run_data['bloom_error_rate'],
            run_data['use_hash'],
            run_data['hash_type'],
            run_data['partial_hash_length'],
        ])

def run_experiment(run_data):
    run_data['start_time'] = datetime.datetime.now().isoformat()

    found_sample = False
    if run_data['use_last_sample'] and os.path.exists('samples/metadata'):
        metadata = get_sample_metadata(run_data)
        with open('samples/metadata', 'r') as f:
            for line in f:
                sample_id, other_metadata = json.loads(line)
                if metadata == other_metadata:
                    print('using older sample_id:')
                    print(sample_id)
                    run_data['sample_id'] = sample_id
                    paths = get_paths(run_data)
                    found_sample = True
                    break

    if not found_sample:
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
            'item original_email',
            'item length',
            'item ratio',
            'item forgotten_email',
            'item frequency_threshold',
            'item emails_generated',
            'item runtime',
            'run start_time',
            'run multiprocess',
            'run use_last_sample',
            'run sample_id',
            'run tokenizer',
            'run ngram_length',
            'run max_email_length',
            'run max_runtime',
            'run use_bins',
            'run bin_size',
            'run bin_bounds',
            'run sample_size',
            'run ratios',
            'run forget_method',
            'run use_frequency_threshold',
            'run use_bloom_filter',
            'run bloom_error_rate',
            'run use_hash',
            'run hash_type',
            'run partial_hash_length'
        ])

    if run_data['use_bins']:
        sample_size = run_data['bin_size'] * len(run_data['bin_bounds'])
    else:
        sample_size = run_data['sample_size']

    if run_data['multiprocess']:
        pool = Pool()
        map_f = functools.partial(pool.imap_unordered, chunksize=10)
    else:
        map_f = map

    with shelve.open(paths['forgotten_sample'], 'r') as sample:
        for ratio in run_data['ratios']:
            i = 0
            last_percentage = 0
            print('processing ratio: {}'.format(ratio))
            items = sample[str(ratio)].values()
            if run_data['use_bloom_filter']:
                with shelve.open(paths['bloom_filters'], 'r') as bloom_filters:
                    for item in items:
                        item['bloom_filter'] = bloom_filters[item['md5']]
            if not run_data['use_frequency_threshold']:
                for item in items:
                    item['frequency_threshold'] = None
            target = functools.partial(recall_email, run_data=run_data)
            processed_items = map_f(target, items)
            for item in processed_items:
                print_to_csv(item, run_data, paths['results'])
                i += 1
                percentage = round(100*i/sample_size)
                if percentage > last_percentage:
                    last_percentage = percentage
                    print('\tprocessed: {}%'.format(percentage))

    if run_data['multiprocess']:
        pool.close()
        pool.join()

def run_all_experiments(base_run_data=None):
    count = 0
    experiments = []
    if base_run_data is None:
        base_run_data = get_run_data({})
    else:
        base_run_data = get_run_data(base_run_data)
    paths = get_paths(base_run_data)
    if not os.path.exists(paths['emails']):
        save_emails(base_run_data)
    sample_id = save_original_sample(base_run_data)

    for use_bloom_filter in (True, False):
        for bloom_error_rate in (0.1, 0.01, 0.001):
            for use_hash in (True, False):
                for forget_method in ('frequency', 'random'):
                    for use_frequency_threshold in (True, False):
                        for hash_type in ('full', 'split', 'independent'):
                            for partial_hash_length in range(10, 60, 10):
                                # skip duplicate experiments
                                if forget_method == 'random' and use_frequency_threshold:
                                    continue
                                if not use_hash and hash_type != 'full':
                                    continue
                                if hash_type != 'split' and partial_hash_length != 10:
                                    continue
                                if not use_bloom_filter and bloom_error_rate != 0.1:
                                    continue
                                count += 1
                                run_data = base_run_data.copy()
                                run_data.update({
                                    'sample_id': sample_id,
                                    'use_bloom_filter': use_bloom_filter,
                                    'bloom_error_rate': bloom_error_rate,
                                    'use_hash': use_hash,
                                    'forget_method': forget_method,
                                    'use_frequency_threshold': use_frequency_threshold,
                                    'hash_type': hash_type,
                                    'partial_hash_length': partial_hash_length,
                                })
                                run_data = get_run_data(run_data)
                                experiments.append(run_data)

    for i, run_data in enumerate(experiments):
        for other_run_data in experiments[i+1:]:
            if run_data == other_run_data:
                raise Exception('Some experiments are duplicate')

    print('running {} experiments'.format(count))
    for run_data in experiments:
        run_experiment(run_data)

def get_run_data(run_data):
    for param, default in DEFAULT_RUN_DATA.items():
        if param not in run_data:
            run_data[param] = default
    for param in run_data:
        if param not in DEFAULT_RUN_DATA:
            raise Exception('{} is not a valid run_data option'.format(param))
    if run_data['use_last_sample']:
        run_data['sample_id'] = None
    if not run_data['use_hash']:
        run_data['hash_type'] = None
    if run_data['hash_type'] != 'split':
        run_data['partial_hash_length'] = None
    if run_data['forget_method'] == 'random':
        run_data['use_frequency_threshold'] = False
    if not run_data['use_bloom_filter']:
        run_data['bloom_error_rate'] = None
    if run_data['use_bins']:
        run_data['sample_size'] = None
        run_data['max_email_length'] = None
    else:
        run_data['bin_bounds'] = None
        run_data['bin_size'] = None
    return run_data

DEFAULT_RUN_DATA = {
    'multiprocess': False,
    'use_last_sample': True,
    'sample_id': None,
    'tokenizer': 'moses',
    'ngram_length': 3,
    # 198185 (=81%) emails lower than or equal to 500 in length
    'max_email_length': 500,
    'max_runtime': 1000,
    'use_bins': False,
    'bin_size': 100,
    'bin_bounds': [[i,i+50] for i in range(50, 550, 50)],
    'sample_size': 1000,
    'ratios': [round(0.1 * i, 1) for i in range(1, 6)],
    # random | frequency
    'forget_method': 'frequency',
    'use_frequency_threshold': True,
    'use_bloom_filter': True,
    'bloom_error_rate': 0.01,
    'use_hash': False,
    # full | split | independent
    'hash_type': 'split',
    'partial_hash_length': 10,
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('run_data', nargs='?', type=json.loads)
    parser.set_defaults(run_data=DEFAULT_RUN_DATA)
    run_data = vars(parser.parse_args())['run_data']
    run_data = get_run_data(run_data)
    run_experiment(run_data)
