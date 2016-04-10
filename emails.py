import os, email, pickle, marisa_trie, random, csv, hashlib, shelve, itertools, functools
from collections import Counter, defaultdict
from multiprocessing import Pool
from threading import Lock
from pprint import PrettyPrinter

#for reproduceability
random.seed(0)
email_count_lock = Lock()
email_count = 0

def preprocess():
    ngram_counter = defaultdict(lambda: 0)
    #list of all valid email paths
    paths = (f[0]+'/'+g for f in os.walk('maildir/') for g in f[2])
    paths = filter(os.path.isfile, paths)

    with Pool() as pool:
        #stringify email parts
        tmp = pool.imap_unordered(email_to_str_list, paths) 
        #filter unparseable emails
        tmp = (p for m in filter(lambda m: m is not None, tmp) for p in m)
        #tokenize and count ngrams
        tmp = pool.imap_unordered(tokenize_email, tmp) 
        print('parsing emails...')
        with shelve.open('emails.shelve') as emails:
            for i, t in enumerate(tmp):
                if t[0] not in emails:
                    emails[t[0]] = t[1]
                for ngram, c in iter(count_ngrams(t[1]).items()):
                    ngram_counter[ngram] += c
                if i%1000 == 0: 
                    print('processing email...', i)

    print('saved dict of email tokens hashed by md5 in emails.shelve')

    print('building trie...')
    trie = marisa_trie.RecordTrie('I', 
        ((k, (v,)) for k, v in iter(ngram_counter.items())))
    print('saving trie...')
    trie.save('trie.marisa')
    print('saved marisa_trie.RecordTrie of ngram frequency hashed by ngram in trie.marisa''')

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

def find_ngrams(partial_ngram, trie, n=3):
    global email_count
    if None in partial_ngram:
        split = partial_ngram.index(None)
        prefix = partial_ngram[:split]
        constraints = partial_ngram[split:n]
        if len(prefix) > 0:
            ngrams = trie.iteritems(' '.join(prefix)+' ')
        else:
            ngrams = trie.iteritems()
        ngrams = filter(lambda item: 
            len(item[0].split(' '))==n and
            all(item[0].split(' ')[split:][i] == c 
                for i, c in enumerate(constraints) if c is not None)
            , ngrams)
        ngrams = list(ngrams)
        ngrams.sort(key=lambda item: item[1][0], reverse=True)
        email_count_lock.acquire()
        email_count+=len(ngrams)
        email_count_lock.release()
        return [ngram.split(' ') for ngram, v in ngrams]

def make_emails(tokens, trie, length=None, n=3):
    global email_count
    email_count_lock.acquire()
    email_count+=1
    email_count_lock.release()
    if not length:
        length = len(tokens)

    if None not in tokens:
        yield tokens

    elif len(tokens) == n:
        for ngram in find_ngrams(tokens, trie):
            yield ngram 

    elif None not in tokens[:-1] and tokens[-1] is None:
        for ngram in make_emails(tokens[-n:], trie, length):
            yield tokens[:-n]+ngram

    else:
        for msg in make_emails(tokens[:-1], trie, length):
            msg.append(tokens[-1])
            for msg in make_emails(msg, trie, length):
                yield msg

def forget_email(tokens, ratio):
    l = len(tokens)
    forget = random.sample(range(l), int((1-ratio)*l))
    return tuple(token if i in forget else None for i, token in enumerate(tokens))

def recall_email(tokens, trie, md5):
    global email_count
    result = {}
    result['hashed_count'] = 0
    for msg in make_emails(list(tokens), trie):
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

def email_stats(msg, ratio, skip=False):
    result = {}
    md5, msg = msg
    tokens = forget_email(msg, ratio)
    result['md5'] = md5
    result['length'] = len(msg)
    result['ratio'] = "{0:.1f}".format(ratio)
    if skip and not tokens[0]:
        return result
    result['data'] = recall_email(tokens, trie, md5)
    return result

def main(sample_size=100, ratio_step=0.1, max_len=50, verbose=False):
    global trie
    n = 3
    print('opening emails and trie...')
    with shelve.open('emails.shelve') as emails:
        trie = marisa_trie.RecordTrie('I')
        trie.load('trie.marisa')
        print('gathering stats...')
        with open('stats.csv', 'w') as f:
            csv.writer(f).writerow([
                'md5',
                'number of tokens',
                'ratio forgotten',
                'email tree edges computed',
                'complete emails compared',
            ])
        
        steps = int(1/ratio_step)
        ratios = (step*ratio_step for step in range(1,steps))
        for ratio in ratios:
            with Pool() as pool:
                results = pool.imap_unordered(
                    functools.partial(email_stats, ratio=ratio), 
                    ((h, msg) for h, msg in iter(emails.items()) if n<=len(msg)<= max_len))
                for r in itertools.islice((r for r in results if 'data' in r), sample_size):
                    with open('stats.csv', 'a') as f:
                        if verbose: PrettyPrinter().pprint(r)
                        csv.writer(f).writerow([
                            r['md5'],
                            r['length'],
                            r['ratio'],
                            r['data']['email_count'],
                            r['data']['hashed_count'],
                        ])
if __name__ == '__main__':
    main()