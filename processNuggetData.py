import numpy as np
import cPickle
from collections import defaultdict, OrderedDict
import sys, re
import pandas as pd
import random

#thien's version
fetCutoff = 2
window = 31
useEligible = False

def lookup(mess, key, gdict, addOne):
    if key not in gdict:
        nk = len(gdict)
        if addOne: nk += 1
        gdict[key] = nk
        if mess: print mess, ': ', key, ' --> id = ', gdict[key]

def build_data(srcDir, dataCorpus):
    """
    Loads data.
    """
    
    revs = {}
    
    corpusCountIns = defaultdict(int)
    corpusCountTypes = defaultdict(lambda: defaultdict(int))
    corpusCountSubTypes = defaultdict(lambda: defaultdict(int))
    corpusCountRealis = defaultdict(lambda: defaultdict(int))
    corpusCountCoref = defaultdict(int)
    maxLength = -1
    lengthCounter = defaultdict(int)
    
    currentDoc = ''
    sdict = defaultdict(list)
    ddict = {}
    
    mpdict = {}
    mpdict['pos'] = {'######':0}
    mpdict['chunk'] = {'######':0,'O':1}
    mpdict['clause'] = {'######':0}
    mpdict['possibleTypes'] = {'NONE':0}
    mpdict['dep'] = {'NONE':0}
    mpdict['nonref'] = {'######':0,'false':1}
    mpdict['title'] = {'######':0,'false':1}
    mpdict['eligible'] = {'######':0,'0':1}
    mpdict['type'] = {'NONE':0}
    mpdict['subtype'] = {'NONE':0}
    mpdict['realis'] = {'NONE':0}
    
    vocab = defaultdict(int)
    nodeFetCounter = defaultdict(int)
    for _data in dataCorpus:
        revs[_data] = {}
        with open(srcDir + '/' + _dat + '.txt', 'r') as f:
            for line in f:
                line = line.strip()
                
                if line.startswith('#BeginOfDocument'):
                    currentDoc = line[(line.find(' ')+1):]
                    revs[_data][currentDoc] = {'instances' : [], 'coreference' : []}
                    continue
                
                if line == '#EndOfDocument':
                    currentDoc = ''
                    continue
                
                if not line and not currentDoc:
                    continue
                
                if not line:
                    length = len(sdict['token'])
                    lengthCounter[length] += 1
                    if length > maxLength: maxLength = length
                    for anchorIndex in range(length):
                        inst = parseInst(sdict, ddict, anchorIndex, window, useEligible)
                        revs[_data][currentDoc]['instances'] += [inst]
                        updateCounters(_data, inst, corpusCountIns, corpusCountTypes, corpusCountSubTypes, corpusCountRealis)
                    sdict = defaultdict(list)
                    continue
                
                if line.startswith('@Coreference'):
                    revs[_data][currentDoc]['coreference'] += [parseCoreferenceLine(line)]
                    corpusCountCoref[_data] += len(revs[_data][currentDoc]['coreference'])
                    continue
                
                parseLine(line, sdict, ddict, mpdict, vocab, nodeFetCounter if 'train' in _data else None)
                
    
    print 'instances in corpus'
    for corpus in corpusCountIns:
    	print corpus, ' : ', corpusCountIns[corpus]
    print '---------------'
    print 'length distribution'
    for le in lengthCounter:
    	print le, ' : ', lengthCounter[le]
    print '---------------'
    print "maximum length of sentences: ", maxLength
    
    itypeDict = {}
    for et in mpdict['type']: itypeDict[mpdict['type'][et]] = et
    isubtypeDict = {}
    for est in mpdict['subtype']: isubtypeDict[mpdict['subtype'][est]] = est
    irealisDict = {}
    for ere in mpdict['realis']: irealisDict[mpdict['realis'][ere]] = ere
    
    for cop in corpusCountIns:
        print '----------------%s--------------', cop
        print '#instances: ', corpusCountIns[cop]
        print '#coreference: ', corpusCountCoref[cop]
        displayStats('type', corpusCountTypes[cop], itypeDict)
        displayStats('subtype', corpusCountSubTypes[cop], isubtypeDict)
        displayStats('realis', corpusCountRealis[cop], irealisDict)
        
    return revs, mpdict, vocab, nodeFetCounter
    
def displayStats(mess, stats, idict):
    print '>>>>%s<<<<', mess
    for v in stats:
        print '#' + idict[v], ' : ', stats[v]

def updateCounters(_data, inst, corpusCountIns, corpusCountTypes, corpusCountSubTypes, corpusCountRealis):
    etype = inst['type']
    esubtype = inst['subtype']
    erealis = inst['realis']
    
    corpusCountIns[_data] += 1
    corpusCountTypes[_data][etype] += 1
    corpusCountSubTypes[_data][esubtype] += 1
    corpusCountRealis[_data][erealis] += 1

def parseCoreferenceLine(line):
    els = line.split('\t')
    
    if len(els) != 3:
        print 'coreference chain does not have length 3: ', line
        exit()
        
    chain = els[2].split(',')
    return chain

def parseLine(line, sdict, ddict, mpdict, vocab, nodeFetCounter):
    els = line.split('\t')
                
    if len(els) != 21:
        print 'incorrect line format: ', line
            exit()
                
    tokenId = int(els[0])
    #sdict['tokenId'] += [tokenId]
    tokenStart = int(els[1])
    sdict['tokenStart'] += [tokenStart]
    tokenEnd = int(els[2])
    sdict['tokenEnd'] += [tokenEnd]
                
    token = els[3]
    sdict['token'] += [token]
    vocab[token] += 1
    if 'token' not in ddict: ddict['token'] = '######'
    
    lemma = els[4]
    #sdict['lemma'] += [lemma]
    
    pos = els[5]
    lookup('POS', pos, mpdict['pos'], False)
    sdict['pos'] += [mpdict['pos'][pos]]
    if 'pos' not in ddict: ddict['pos'] = 0
    
    chunk = els[6]
    lookup('CHUNK', chunk, mpdict['chunk'], False)
    sdict['chunk'] += [mpdict['chunk'][chunk]]
    if 'chunk' not in ddict: ddict['chunk'] = 0
    
    nomlex = els[7]
    #sdict['nomlex'] += [nomlex]
    
    clause = els[8]
    if clause not in mpdict['clause']:
        mpdict['clause'][clause] = int(clause) + 1
        print 'CLAUSE: ', clause, ' id --> ', mpdict['clause'][clause]
    sdict['clause'] += [mpdict['clause'][clause]]
    if 'clause' not in ddict: ddict['clause'] = 0
    
    possibleTypes = els[9].split()
    for piptype in possibleTypes:
        lookup('POSSIBLE TYPE', piptype, mpdict['possibleTypes'], False)
    sdict['possibleTypes'] += [ [mpdict['possibleTypes'][piptype] for piptype in possibleTypes] ]
    if 'possibleTypes' not in ddict: ddict['possibleTypes'] = []
    
    synonyms = els[10].split()
    #sdict['synonyms'] += [synonyms]
    
    browns = els[11].split()
    #sdict['browns'] += [browns]
    
    dep = els[12].split()
    lookup('DEP', dep, mpdict['dep'], False)
    sdict['dep'] += [mpdict['dep'][dep]]
    if 'dep' not in ddict: ddict['dep'] = []
    
    nonref = els[13]
    lookup('NONREF', nonref, mpdict['nonref'], False)
    sdict['nonref'] += [mpdict['nonref'][nonref]]
    if 'nonref' not in ddict: ddict['nonref'] = 0
    
    title = els[14]
    lookup('TITLE', title, mpdict['title'], False)
    sdict['title'] += [mpdict['title'][title]]
    if 'title' not in ddict: ddict['title'] = 0
    
    eligible = els[15]
    lookup('ELIGIBLE', eligible, mpdict['eligible'], False)
    sdict['eligible'] += [mpdict['eligible'][eligible]]
    if 'eligible' not in ddict: ddict['eligible'] = 0
    
    sparseFeatures = els[16].split()
    sdict['sparseFeatures'] += [sparseFeatures]
    if nodeFetCounter:
        for sps in sparseFeatures: nodeFetCounter[sps] += 1
    
    etype = els[17]
    lookup('EVENT TYPE', etype, mpdict['type'], False)
    sdict['type'] += [mpdict['type'][etype]]
    
    esubtype = els[18]
    lookup('EVENT SUBTYPE', esubtype, mpdict['subtype'], False)
    sdict['subtype'] += [mpdict['subtype'][esubtype]]
    
    erealis = els[19]
    lookup('EVENT REALIS', erealis, mpdict['realis'], False)
    sdict['realis'] += [mpdict['realis'][erealis]]
    
    eeventId = els[20]
    sdict['eventId'] += [eeventId]

def parseInst(sdict, ddict, anchorIndex, window, useEligible=False):

    length = len(sdict['token'])
    if anchorIndex < 0 or anchorIndex >= length: return None
    
    if sdict['eligible'][anchorIndex] == 1 and useEligible: return None
    
    inst = {}
    
    lower = anchorIndex - window / 2
    upper = lower + window - 1
    
    newAnchorIndex = anchorIndex - lower
    
    for i in range(window):
        id = i + lower
        for key in ddict:
            addent = ddict[key]
            if id >= 0 and id < length: addent = sdict[key][id]
            if key not in inst: inst[key] = []
            inst[key] += [addent]
    
    for key in sdict:
        if key not in ddict:
            inst[key] = sdict[key][anchorIndex]
    
    inst['anchor'] = newAnchorIndex
    
    return inst

def get_W(word_vecs, k=300):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k))
    W[0] = np.zeros(k)
    word_idx_map['######'] = 0
    i = 1
    for word in word_vecs:
        W[i] = word_vecs[word]
        word_idx_map[word] = i
        i += 1
    return W, word_idx_map

def load_bin_vec(fname, vocab):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    dim = 0
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)   
            if word in vocab:
               word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
               dim = word_vecs[word].shape[0]
            else:
                f.read(binary_len)
    print 'dim: ', dim
    return dim, word_vecs
    
def load_text_vec(fname, vocab):
    word_vecs = {}
    count = 0
    dim = 0
    with open(fname, 'r') as f:
        for line in f:
            count += 1
            line = line.strip()
            if count == 1:
                if len(line.split()) < 10:
                    dim = int(line.split()[1])
                    print 'dim: ', dim
                    continue
                else:
                    dim = len(line.split()) - 1
                    print 'dim: ', dim
            word = line.split()[0]
            emStr = line[(line.find(' ')+1):]
            if word in vocab:
                word_vecs[word] = np.fromstring(emStr, dtype='float32', sep=' ')
                if word_vecs[word].shape[0] != dim:
                    print 'mismatch dimensions: ', dim, word_vecs[word].shape[0]
                    exit()
    print 'dim: ', dim
    return dim, word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=300):
    """
    For words that occur in at least min_df documents, create a separate word vector.    
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

if __name__=="__main__":
    np.random.seed(8989)
    random.seed(8989)
    embType = sys.argv[1]
    w2v_file = sys.argv[2]
    srcDir = sys.argv[3]
    
    dataCorpus = ["train", "valid", "test"]
    
    print "loading data...\n"
    revs, mpdict, vocab, nodeFetCounter = build_data(srcDir, dataCorpus)
    
    print "data loaded!"
    print "vocab size: " + str(len(vocab))
    print "loading word embeddings...",
    dimEmb = 300
    if embType == 'word2vec':
    	dimEmb, w2v = load_bin_vec(w2v_file, vocab)
    else:
    	dimEmb, w2v = load_text_vec(w2v_file, vocab)
    print "word embeddings loaded!"
    print "num words already in word embeddings: " + str(len(w2v))
    add_unknown_words(w2v, vocab, 1, dimEmb)
    W1, word_idx_map = get_W(w2v, dimEmb)
    rand_vecs = {}
    add_unknown_words(rand_vecs, vocab, 1, dimEmb)
    W2, _ = get_W(rand_vecs, dimEmb)
    
    maxLength = window
    dist_size = 2*maxLength - 1
    dist_dim = 50
    D = np.random.uniform(-0.25,0.25,(dist_size+1,dist_dim))
    D[0] = np.zeros(dist_dim)
    
    pos_dim = 50
    POS = np.random.uniform(-0.25,0.25,(len(mpdict['pos']),pos_dim))
    POS[0] = np.zeros(pos_dim)
    
    chunk_dim = 50
    CHUNK = np.random.uniform(-0.25,0.25,(len(mpdict['chunk']),chunk_dim))
    CHUNK[0] = np.zeros(chunk_dim)
    
    clause_dim = 50
    CLAUSE = np.random.uniform(-0.25,0.25,(len(mpdict['clause']),clause_dim))
    CLAUSE[0] = np.zeros(clause_dim)
    
    nonref_dim = 50
    NONREF = np.random.uniform(-0.25,0.25,(len(mpdict['nonref']),nonref_dim))
    NONREF[0] = np.zeros(nonref_dim)
    
    title_dim = 50
    TITLE = np.random.uniform(-0.25,0.25,(len(mpdict['title']),title_dim))
    TITLE[0] = np.zeros(title_dim)
    
    eligible_dim = 50
    ELIGIBLE = np.random.uniform(-0.25,0.25,(len(mpdict['eligible']),eligible_dim))
    ELIGIBLE[0] = np.zeros(eligible_dim)
    
    embeddings = {}
    embeddings['word'] = W1
    embeddings['randomWord'] = W2
    embeddings['anchor'] = D
    embeddings['pos'] = POS
    embeddings['chunk'] = CHUNK
    embeddings['clause'] = CLAUSE
    embeddings['nonref'] = NONREF
    embeddings['title'] = TITLE
    embeddings['eligible'] = ELIGIBLE
    
    for di in mpdict:
        print 'size of ', di, ': ', len(mpdict[di])
    
    print 'dumping ...'
    cPickle.dump([revs, embeddings, mpdict], open('win_' + str(window) + '.' + embType + "_nugget.pkl", "wb"))
    print "dataset created!"   
