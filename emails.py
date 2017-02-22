import os, email, marisa_trie, random, csv, hashlib, shelve, itertools, functools, re, pybloomfilter, shutil, datetime
from collections import Counter, defaultdict
from multiprocessing import Pool
from threading import Lock
from pprint import PrettyPrinter

#for reproduceability
random.seed(0)
email_count_lock = Lock()
email_count = 0

def preprocess(regex=True, bloom_filter=True):
    shelve_path, trie_path = get_paths(regex, bloom_filter)
    ngram_counter = defaultdict(lambda: 0)
    #list of all valid email paths
    paths = (f[0]+'/'+g for f in os.walk('maildir/') for g in f[2])
    paths = filter(os.path.isfile, paths)
    #only process 10 emails
    paths = itertools.islice(paths, 10)
    with Pool() as pool:
        #stringify email parts
        print('parsing emails...')
        tmp = pool.imap_unordered(email_to_str_list, paths) 
        #filter unparseable emails
        tmp = (p for m in filter(lambda m: m is not None, tmp) for p in m)
        #tokenize and count ngrams
        print('tokenizing emails...')
        if regex:
            tmp = pool.imap_unordered(tokenize_email_regex, tmp)
        else:
            tmp = pool.imap_unordered(tokenize_email, tmp) 
        print('processing emails...')
        with shelve.open(shelve_path) as emails:
            for i, (tokens_hash, tokens) in enumerate(tmp):
                if tokens_hash not in emails:
                    if bloom_filter:
                        bloom_path = 'bloom_'+tokens_hash
                        bloom = pybloomfilter.BloomFilter(len(tokens), 0.1, bloom_path)
                        bloom.update(tokens)
                        bloom.sync()
                        with open(bloom_path, 'rb') as bloom_file:
                            emails[tokens_hash] = (tokens, bloom_file.read())
                        os.remove(bloom_path)
                    else:
                        emails[tokens_hash] = tokens
                for ngram, c in iter(count_ngrams(tokens).items()):
                    ngram_counter[ngram] += c
                if i%1000 == 0: 
                    print('processing email:', i)

    print('saved emails')

    print('building trie...')
    trie = marisa_trie.RecordTrie('I', 
        ((k, (v,)) for k, v in iter(ngram_counter.items())),order=marisa_trie.WEIGHT_ORDER)
    print('saving trie...')
    trie.save(trie_path)
    print('saved trie')

def get_paths(regex, bloom_filter):
    if regex:
        if bloom_filter:
            shelve_path = 'emails_regex_bloom.shelve'
        else:
            shelve_path = 'emails_regex.shelve'
        trie_path = 'trie_regex.marisa'
    else:
        if bloom_filter:
            shelve_path = 'emails_bloom.shelve'
        else:
            shelve_path = 'emails.shelve'
        trie_path = 'trie.marisa'
    return shelve_path, trie_path

def md5_hash(tokens):
    return hashlib.md5(' '.join(tokens).encode('utf-8')).hexdigest()

def email_to_str_list(path):
    with open(path) as m:
        try:
            msg = email.message_from_file(m)
            parts = filter(lambda m: not m.is_multipart(), msg.walk())
            return [part.get_payload() for part in parts]
        except UnicodeDecodeError as e:
            print('cannot parse:', path)

def tokenize_email(msg):
    #nltk word_tokenize was not easily reversible, str.split instead
    tokens = msg.split(' ')
    return (md5_hash(tokens), tokens)

def tokenize_email_regex(msg):
    #only split on whitespace for reversability
    tokens = re.split('\s+', msg)
    return (md5_hash(tokens), tokens)

def count_ngrams(tokens):
    c = defaultdict(lambda: 0)
    for ngram in tokens_to_ngrams(tokens): 
        c[' '.join(ngram)] += 1
    return c

def tokens_to_ngrams(tokens, n=3):
    for i in range(len(tokens)-n+1):
        ngram = tokens[i:i+n]
        for j in range(n):
            yield ngram[:j+1]

def find_ngrams(partial_ngram, trie, bloom_filter, n=3):
    global email_count
    if None in partial_ngram:
        split = partial_ngram.index(None)
        prefix = partial_ngram[:split]
        suffix = partial_ngram[split:n]
        if len(prefix) > 0:
            ngrams = trie.iteritems(' '.join(prefix)+' ')
        else:
            ngrams = trie.iteritems()
        ngrams = filter(lambda item: 
            len(item[0].split(' '))==n and
            all(item[0].split(' ')[split:][i] == c 
                for i, c in enumerate(suffix) if c is not None)
            , ngrams)
        if bloom_filter:
            ngrams = filter(lambda item: all(word in bloom_filter for word in item[0].split(' ')), ngrams)
        ngrams = sorted(ngrams, key=lambda item: item[1][0], reverse=True)
        email_count_lock.acquire()
        email_count+=len(ngrams)
        email_count_lock.release()
        return (ngram.split(' ') for ngram, v in ngrams)

def make_emails(tokens, trie, bloom_filter, length=None, n=3):
    global email_count
    email_count_lock.acquire()
    email_count+=1
    email_count_lock.release()
    if not length:
        length = len(tokens)

    if None not in tokens:
        yield tokens

    elif len(tokens) == n:
        for ngram in find_ngrams(tokens, trie, bloom_filter):
            yield ngram 

    elif None not in tokens[:-1] and tokens[-1] is None:
        for ngram in make_emails(tokens[-n:], trie, bloom_filter, length):
            yield tokens[:-n]+ngram

    else:
        for msg in make_emails(tokens[:-1], trie, bloom_filter, length):
            msg.append(tokens[-1])
            for msg in make_emails(msg, trie, bloom_filter, length):
                yield msg

def forget_email(tokens, ratio):
    l = len(tokens)
    forget = random.sample(range(l), int((1-ratio)*l))
    return tuple(token if i in forget else None for i, token in enumerate(tokens))

def recall_email(tokens, trie, md5, bloom_filter):
    global email_count
    result = {}
    result['hashed_count'] = 0
    for msg in make_emails(list(tokens), trie, bloom_filter):
        result['hashed_count'] += 1
        try:
            if md5_hash(msg) == md5:
                # result['msg'] = msg 
                email_count_lock.acquire()
                result['email_count'] = email_count
                email_count = 0
                email_count_lock.release()
                return result
        except TypeError as e:
            print(tokens, msg)
            raise e
    raise Exception()

def email_stats(item, ratio, bloom_filter, skip=False):
    result = {}
    if bloom_filter:
        md5 = item[0]
        msg = item[1]
        bloom_path = 'tmp/bloom_'+md5
        with open(bloom_path, 'wb') as bloom_file:
            bloom_file.write(item[2])
        bloom = pybloomfilter.BloomFilter.open(bloom_path)
    else:
        md5, msg = item
        bloom = None
    tokens = forget_email(msg, ratio)
    result['md5'] = md5
    result['length'] = len(msg)
    result['ratio'] = "{0:.1f}".format(ratio)
    if skip and not tokens[0]:
        return result
    result['data'] = recall_email(tokens, trie, md5, bloom)
    if bloom_filter:
        os.remove(bloom_path)
    return result

def main(sample_size=50, ratio_step=0.1, max_ratio=0.5, max_len=50, verbose=False, bloom_filter=True, regex=True, n=3):
    global trie
    os.mkdir('tmp')
    shelve_path, trie_path = get_paths(regex, bloom_filter)
    print('opening trie...')
    trie = marisa_trie.RecordTrie('I')
    trie.load(trie_path)
    time = datetime.datetime.now().isoformat()
    if not os.path.isfile('stats.csv'):
        with open('stats.csv', 'w') as f:
            csv.writer(f).writerow([
                'md5',
                'number of tokens',
                'ratio forgotten',
                'email tree edges computed',
                'complete emails compared',
                'run info',
            ])
    with shelve.open(shelve_path) as emails:
        print('gathering stats...')
        steps = int(1/ratio_step)
        ratios = (step*ratio_step for step in range(1,steps))
        for ratio in ratios:
            with Pool() as pool:
                if bloom_filter:
                    items = ((h, item[0], item[1]) for h, item in iter(emails.items()) if n<=len(item[0])<= max_len)
                else:
                    items = ((h, item) for h, item in iter(emails.items()) if n<=len(item)<= max_len)
                results = pool.imap_unordered(
                    functools.partial(email_stats, ratio=ratio, bloom_filter=bloom_filter), items)
                for r in itertools.islice((r for r in results if 'data' in r), sample_size):
                    with open('stats.csv', 'a') as f:
                        if verbose: PrettyPrinter().pprint(r)
                        csv.writer(f).writerow([
                            r['md5'],
                            r['length'],
                            r['ratio'],
                            r['data']['email_count'],
                            r['data']['hashed_count'],
                            'time: {}, sample_size: {}, max_len: {}, bloom_filter: {}, regex: {}, n: {}'.format(time, sample_size, max_len, bloom_filter, regex, n),
                        ])
    print('cleaning up bloom filter files...')
    shutil.rmtree('tmp')

if __name__ == '__main__':
    main()
