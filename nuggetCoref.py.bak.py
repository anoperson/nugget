import numpy
import time
import sys
import subprocess
import os
import random
import cPickle
import copy

import theano
from theano import tensor as T
from collections import OrderedDict, defaultdict
from theano.tensor.nnet import conv
from theano.tensor.signal import downsample
import theano.tensor.shared_randomstreams
from modelCoref import *

goldFiles = OrderedDict([('train', ''),
                         ('valid', '/scratch/thn235/projects/nugget/golds/hopper/eval.tbf'),
                         ('test', '')
                        ])
tokenFiles = OrderedDict([('train', ''),
                          ('valid', '/scratch/thn235/projects/nugget/golds/hopper/tkn/'),
                          ('test', '')
                         ])

scoreScript = '/scratch/thn235/projects/nugget/scorer/scorer_v1.7.py'
conllTempFile = '/scratch/thn235/projects/nugget/scorer/conllTempFile_Coreference.txt'
subtype2typeMap = {"declarebankruptcy": "business",

                   "artifact": "manufacture",
                   
                   "startposition": "personnel",
                   "endposition": "personnel",
                   "nominate": "personnel",
                   "elect": "personnel",
                   
                   "demonstrate": "conflict",
                   "attack": "conflict",
                   
                   "broadcast": "contact",
                   "contact": "contact",
                   "correspondence": "contact",
                   "meet": "contact",
                   
                   "transfermoney": "transaction",
                   "transferownership": "transaction",
                   "transaction": "transaction",
                   
                   "transportartifact": "movement",
                   "transportperson": "movement",
                   
                   "startorg": "business",
                   "endorg": "business",
                   "mergeorg": "business",
                   
                   "die": "life",
                   "divorce": "life",
                   "marry": "life",
                   "beborn": "life",
                   "injure": "life",
                   
                   "pardon": "justice",
                   "sue": "justice",
                   "convict": "justice",
                   "chargeindict": "justice",
                   "trialhearing": "justice",
                   "sentence": "justice",
                   "appeal": "justice",
                   "releaseparole": "justice",
                   "extradite": "justice",
                   "fine": "justice",
                   "execute": "justice",
                   "arrestjail": "justice",
                   "acquit": "justice"}

##################################################################

def generateDataInstance(rev, dictionaries, embeddings, features, window):
    #assuming fitLen is even, window is odd
    fitLen = len(rev['word'])-1
    if window < fitLen:
        upper = (window + fitLen) / 2
        lower = upper - window + 1
        for prop in rev:
            if isinstance(rev[prop], list):
                rev[prop] = rev[prop][lower:(upper+1)]
            elif prop == 'anchor':
                rev['anchor'] = fitLen / 2 - lower
        embeddings['anchor'] = embeddings['anchor'][0:(2*window)]

    numAnchor = embeddings['anchor'].shape[0]-1
    numPossibleTypes = len(dictionaries['possibleTypes'])
    numDep = len(dictionaries['dep'])
    
    res = defaultdict(list)
    for id in range(len(rev['word'])):
        for fet in features:
            if fet == 'word':
                if rev['word'][id] not in dictionaries['word']:
                    print 'cannot find id for word: ', rev['word'][id]
                    exit()
                res['word'] += [ dictionaries['word'][rev['word'][id]] ]
                continue
            
            if fet == 'anchor':
                anchor = numAnchor / 2 + id - rev['anchor']
                scalar_anchor = anchor+1
                vector_anchor = [0] * numAnchor
                vector_anchor[anchor] = 1
                res['anchor'].append(vector_anchor if features['anchor'] == 1 else scalar_anchor)
                continue
            
            if fet == 'possibleTypes' or fet == 'dep':
                vector_fet = [0] * (numDep if fet == 'dep' else numPossibleTypes)
                for fid in rev[fet][id]:
                    vector_fet[fid] = 1
                res[fet].append(vector_fet)
                continue
            
            numFet = len(dictionaries[fet])-1
            scalar_fet = rev[fet][id]
            vector_fet = [0] * numFet
            if scalar_fet > 0:
                vector_fet[scalar_fet-1] = 1
            res[fet].append(vector_fet if features[fet] == 1 else scalar_fet)
    
    return res

def make_data(revs, dictionaries, embeddings, features1, features2, window, eventTypePath):

    mLen = -1
    for datn in revs:
        for doc in revs[datn]:
            for ins in revs[datn][doc]['instances']:
                if len(ins['word']) > mLen:
                    mLen = len(ins['word'])
    
    print 'maximum length in th original dataset: ', mLen
    print 'using window for instances: ', window
    
    typeDict = dict((k,v) for v,k in dictionaries['type'].iteritems())
    subtypeDict = dict((k,v) for v,k in dictionaries['subtype'].iteritems())
    realisDict = dict((k,v) for v,k in dictionaries['realis'].iteritems())
    
    eventTypeStorer, realisStorer, wholeLineEventTypeRealisStorer, eventIdStorer, corefChainStorer = readEventTypeFile(eventTypePath)
    
    res = {}
    idMappings = {}
    coutPositive = 0
    for datn in revs:
        res[datn] = defaultdict(list)
        idMappings[datn] = {}
        iid = -1
        for doc in revs[datn]:
            for instanceId1 in range(len(revs[datn][doc]['instances'])):
            
                rev1 = revs[datn][doc]['instances'][instanceId1]
                ikey1 = datn + ' ' + doc + ' ' + str(instanceId1)
                
                if ikey1 not in eventTypeStorer: continue
                if (datn == 'train') and (rev1['type'] == 0 or rev1['subtype'] == 0 or rev1['realis'] == -1):
                    print 'should be an event istance 1 here, but label from original data does not show that!'
                    exit()
                    
                if datn == 'train' and realisDict[rev1['realis']] != realisStorer[ikey1]:
                    print 'realis1 mismatched in make data: ', datn, realisDict[rev1['realis']], ' vs ', realisStorer[ikey1]
                    exit()
                
                fetType1, fetSubtype1 = parseTypeSubType(eventTypeStorer[ikey1], rev1, datn, typeDict, subtypeDict)
            
                ists1 = generateDataInstance(rev1, dictionaries, embeddings, features1, window)
                
                eventId1 = rev1['eventId']
                sentId1 = rev1['sentenceId']
                realis1 = realisStorer[ikey1]
                
                for instanceId2 in range(instanceId1+1, len(revs[datn][doc]['instances'])):
                    
                    rev2 = revs[datn][doc]['instances'][instanceId2]
                    ikey2 = datn + ' ' + doc + ' ' + str(instanceId2)
                    
                    if ikey2 not in eventTypeStorer: continue
                    if (datn == 'train') and (rev2['type'] == 0 or rev2['subtype'] == 0 or rev2['realis'] == -1):
                        print 'should be an event istance 2 here, but label from original data does not show that!'
                        exit()
                    
                    if datn == 'train' and realisDict[rev2['realis']] != realisStorer[ikey2]:
                        print 'realis2 mismatched in make data: ', datn, realisDict[rev2['realis']], ' vs ', realisStorer[ikey2]
                        exit()
                        
                    fetType2, fetSubtype2 = parseTypeSubType(eventTypeStorer[ikey2], rev2, datn, typeDict, subtypeDict)
            
                    ists2 = generateDataInstance(rev2, dictionaries, embeddings, features2, window)
                    
                    eventId2 = rev2['eventId']
                    sentId2 = rev2['sentenceId']
                    realis2 = realisStorer[ikey2]
                    
                    #coreference features
                    typeMatch = 'TypeMatch=' + ('true' if fetType1 == fetType2 else 'false')
                    subtypeMatch = 'SubTypeMatch=' + ('true' if fetSubtype1 == fetSubtype2 else 'false')
                    realisMatch = 'RealisMatch=' + ('true' if realis1 == realis2 else 'false')
                    sentDistance = 'SentDist=' + (str(sentId1-sentId2) if sentId1 > sentId2 else str(sentId2-sentId1))
                
                    coreferenceFeatures = [typeMatch, subtypeMatch, realisMatch, sentDistance]
                    ######
                
                    for kk in ists1: res[datn][kk + '1'] += [ists1[kk]]
                    for kk in ists2: res[datn][kk + '2'] += [ists2[kk]]
                    
                    res[datn]['binaryFeatures1'] += [rev1['binaryFeatures']]
                    res[datn]['binaryFeatures2'] += [rev2['binaryFeatures']]
                    
                    res[datn]['position1'] += [rev1['anchor']]
                    res[datn]['position2'] += [rev2['anchor']]
                    
                    res[datn]['coreferenceFeatures'] += [coreferenceFeatures]
                    
                    coreferenceLabel = 0
                    corefKey = datn + ' ' + doc
                    if corefKey in corefChainStorer and ((eventId1 + ' ' + eventId2) in corefChainStorer[corefKey] or (eventId2 + ' ' + eventId1) in corefChainStorer[corefKey]):
                        coreferenceLabel = 1
                        coutPositive += 1
                    res[datn]['label'] += [coreferenceLabel if datn != 'test' else -1]
                
                
                    iid += 1
                    idMappings[datn][iid] = ikey1 + '\t' + ikey2 + '\t' + eventId1 + '\t' + eventId2
                    res[datn]['id'] += [iid]
    
    return res, idMappings, wholeLineEventTypeRealisStorer
    
def parseTypeSubType(st, rev, datn, typeDict, subtypeDict):
    els = st.split('_')
    fetType, fetSubtype = els[0], els[1]
    
    cpType, cpSubType = typeDict[rev['type']], subtypeDict[rev['subtype']]
    
    if datn == 'train' and fetType != cpType:
        print 'type from realis file and original dataset not matched: ', datn, fetType, ' vs ', cpType
        exit()
    if datn == 'train' and fetSubtype != cpSubType:
        print 'subtype from realis file and original dataset not matched: ', datn, fetSubtype, ' vs ', cpSubType
        exit()
    
    return fetType, fetSubtype
    
def readEventTypeFile(eventTypePath):
    ccps = ['train', 'test', 'valid']
    eventTypeStorer = {}
    realisStorer = {}
    wholeLineEventTypeRealisStorer = OrderedDict()
    eventIdStorer = {}
    corefChainStorer = defaultdict(set)
    print 'reading ids of instances with types for realis ...'
    currentDoc = ''
    for ccp in ccps:
        if ccp not in wholeLineEventTypeRealisStorer:
            wholeLineEventTypeRealisStorer[ccp] = OrderedDict()
        with open(eventTypePath + '/' + ccp + '.coref') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                
                els = line.split('\t')
                
                if line.startswith('@'):
                    corefKey = ccp + ' ' + currentDoc
                    corefChain = els[2].split(',')
                    for cc1 in corefChain:
                        for cc2 in corefChain:
                            if cc2 != cc1: continue
                            corefChainStorer[corefKey].add(cc1 + ' ' + cc2)
                            corefChainStorer[corefKey].add(cc2 + ' ' + cc1)
                    continue
                    
                doc = els[1]
                
                currentDoc = doc
                
                evid = els[2]
            
                #span = els[3]
                st = els[5].replace('-', '').lower()
                
                reli = els[6]
                
                instanceId = els[-1]
                
                key = ccp + ' ' + doc + ' ' + instanceId
                
                eventTypeStorer[key] = st
                realisStorer[key] = reli
                eventIdStorer[key] = evid
                
                if doc not in wholeLineEventTypeRealisStorer[ccp]: 
                    wholeLineEventTypeRealisStorer[ccp][doc] = []
                wholeLineEventTypeRealisStorer[ccp][doc] += [line]
            
    return eventTypeStorer, realisStorer, wholeLineEventTypeRealisStorer, eventIdStorer, corefChainStorer

def makeBinaryDictionary(dat, cutoff=1):
    if cutoff < 0: return None
    print '-------creating binary feature dictionary on the training data--------'
    
    bfdCounter = defaultdict(int)
    for rev in dat['binaryFeatures1']:
        for fet in rev: bfdCounter[fet] += 1
    print 'binary feature cutoff: ', cutoff
    bfd = {}
    for fet in bfdCounter:
        if bfdCounter[fet] >= cutoff:
            if fet not in bfd: bfd[fet] = len(bfd)
    
    print 'size of binary feature dictionary: ', len(bfd)
    
    return bfd
    
def makeCoreferenceFeatureDictionary(dat, cutoff=1):
    if cutoff < 0: return None
    print '-------creating coreference feature dictionary on the training data--------'
    
    bfdCounter = defaultdict(int)
    for rev in dat['coreferenceFeatures']:
        for fet in rev: bfdCounter[fet] += 1
    print 'coreference feature cutoff: ', cutoff
    bfd = {}
    for fet in bfdCounter:
        if bfdCounter[fet] >= cutoff:
            if fet not in bfd: bfd[fet] = len(bfd)
    
    print 'size of coreference feature dictionary: ', len(bfd)
    
    return bfd

def findMaximumBinaryLength(dats):
    
    maxBiLen = -1
    for corpus in dats:
        for rev in dats[corpus]['binaryFeatures1']:
            if len(rev) > maxBiLen: maxBiLen = len(rev)
        for rev in dats[corpus]['binaryFeatures2']:
            if len(rev) > maxBiLen: maxBiLen = len(rev)
    print 'maximum number of binary features: ', maxBiLen
    
    return maxBiLen

def convertBinaryFeatures(dat, maxBiLen, bfd):
    if not bfd:
        for corpus in dat:
            del dat[corpus]['binaryFeatures1']
            del dat[corpus]['binaryFeatures2']
        return -1
    print 'converting binary features to vectors ...'
    for corpus in dat:
        for i in range(len(dat[corpus]['word1'])):
            dat[corpus]['binaryFeatures1'][i] = getBinaryVector(dat[corpus]['binaryFeatures1'][i], maxBiLen, bfd)
            dat[corpus]['binaryFeatures2'][i] = getBinaryVector(dat[corpus]['binaryFeatures2'][i], maxBiLen, bfd)
            
    return len(bfd)

def getBinaryVector(fets, maxBiLen, dic):
    res = [-1] * (maxBiLen + 1)
    id = 0
    for fet in fets:
        if fet in dic:
            id += 1
            res[id] = dic[fet]
            
    res[0] = id
    return res
    
def convertCoreferenceFeatures(dat, bfd):
    if not bfd:
        for corpus in dat:
            del dat[corpus]['coreferenceFeatures']
        return -1
    print 'converting coreference features to vectors ...'
    for corpus in dat:
        for i in range(len(dat[corpus]['word1'])):
            dat[corpus]['coreferenceFeatures'][i] = getCoreferenceFeatureVector(dat[corpus]['coreferenceFeatures'][i], bfd)
            
    return len(bfd)
    
def getCoreferenceFeatureVector(fets, dic):
    res = [0.0] * len(dic)
    for fet in fets:
        if fet in dic:
            res[dic[fet]] = 1.0
    return res

def predict(corpus, batch, reModel, idx2word, features1, features2):
    evaluateCorpus = {}
    extra_data_num = -1
    nsen = corpus['word1'].shape[0]
    if nsen % batch > 0:
        extra_data_num = batch - nsen % batch
        for ed in corpus:  
            extra_data = corpus[ed][:extra_data_num]
            evaluateCorpus[ed] = numpy.append(corpus[ed],extra_data,axis=0)
    else:
        for ed in corpus: 
            evaluateCorpus[ed] = corpus[ed]
        
    numBatch = evaluateCorpus['word1'].shape[0] / batch
    predictions_corpus = numpy.array([], dtype='int32')
    probs_corpus = []
    for i in range(numBatch):
        zippedCorpus = [ evaluateCorpus[ed + '1'][i*batch:(i+1)*batch] for ed in features1 if features1[ed] >= 0 ] + [ evaluateCorpus[ed + '2'][i*batch:(i+1)*batch] for ed in features2 if features2[ed] >= 0 ]
        zippedCorpus += [ evaluateCorpus['position1'][i*batch:(i+1)*batch], evaluateCorpus['position2'][i*batch:(i+1)*batch] ]
        
        if 'coreferenceFeatures' in evaluateCorpus:
            zippedData += [evaluateCorpus['coreferenceFeatures'][i*batch:(i+1)*batch]]
        
        if 'binaryFeatures1' in evaluateCorpus:
            zippedCorpus += [ evaluateCorpus['binaryFeatures1'][i*batch:(i+1)*batch] ]
            zippedCorpus += [ evaluateCorpus['binaryFeatures2'][i*batch:(i+1)*batch] ]
        
        clas, probs = reModel.classify(*zippedCorpus)
        predictions_corpus = numpy.append(predictions_corpus, clas)
        probs_corpus.append(probs)
    
    probs_corpus = numpy.concatenate(probs_corpus, axis=0)
    
    if extra_data_num > 0:
        predictions_corpus = predictions_corpus[0:-extra_data_num]
        probs_corpus = probs_corpus[0:-extra_data_num]

    return predictions_corpus, probs_corpus

def cluster(corpus, predictions, idMapping):

    tclusters = defaultdict(dict)
    tidclusters = defaultdict(int)
    for id, pred in zip(corpus['id'], predictions):
        if pred == 0: continue
        
        if id not in idMapping:
            print 'cannot find id : ', id , ' in mapping'
            exit()
        keys = idMapping[id].split('\t')
        ikey1, ikey2, evid1, evid2 = keys[0].split(), keys[1].split(), keys[2], keys[3]
        
        ccp1 = ikey1[0]
        ccp2 = ikey2[0]
        if ccp1 != ccp2:
            print 'copurs mismatch in clustering: ', ccp1, ccp2
            exit()
        ccp = ccp1
        
        doc1 = ikey1[1]
        doc2 = ikey2[1]
        if doc1 != doc2:
            print 'document mismatch in clustering: ', doc1, doc2
            exit()
        doc = doc1
        
        if evid1 not in tclusters[doc] and evid2 not in tclusters[doc]:
            cuid = tidclusters[doc]
            tclusters[doc][evid1] = cuid
            tclusters[doc][evid2] = cuid
            tidclusters[doc] += 1
            continue
        if evid1 in tclusters[doc] and evid2 not in tclusters[doc]:
            tclusters[doc][evid2] = tclusters[doc][evid1]
            continue
        if evid1 not in tclusters[doc] and evid2 in tclusters[doc]:
            tclusters[doc][evid1] = tclusters[doc][evid2]
            continue
        
        cuid1 = tclusters[doc][evid1]
        cuid2 = tclusters[doc][evid2]
        for evid in tclusters[doc]:
            if tclusters[doc][evid] == cuid2: tclusters[doc][evid] = cuid1
    
    clusters = defaultdict(lambda : defaultdict(set))
    for doc in tclusters:
        for evid in tclusters[doc]:
            clusters[doc][tclusters[doc][evid]].add(evid)
    
    return clusters    

def writeout(corpus, predictions, probs, idMapping, wholeLineEventTypeRealisStorerCP, ofile):

    clusters = cluster(corpus, predictions, idMapping)
    
    writer = open(ofile, 'w')
    for doc in wholeLineEventTypeRealisStorerCP:
        writer.write('#BeginOfDocument ' + doc + '\n')
        for em in wholeLineEventTypeRealisStorerCP[doc]:
            em = em.split('\t')[0:-1]
            writer.write('\t'.join(em) + '\n')
        if doc in clusters:
            cuid = 0
            for ocid in clusters[doc]:
                oneCluster = list(clusters[doc][ocid])
                writer.write('@Coreference' + '\t' + 'C' + str(cuid) + '\t' + ','.join(oneCluster) + '\n')
                cuid += 1
        writer.write('#EndOfDocument' + '\n')
    writer.close()

def myScore(goldFile, systemFile, tokenFile, conllTempFile):

    proc = subprocess.Popen(["python", scoreScript, "-g", goldFile, "-s", systemFile, "-t", tokenFile, "-c", conllTempFile], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ous, _ = proc.communicate()
    
    spanP, spanR, spanF1 = 0.0, 0.0, 0.0
    subtypeP, subtypeR, subtypeF1 = 0.0, 0.0, 0.0
    realisP, realisR, realisF1 = 0.0, 0.0, 0.0
    realisAndTypeP, realisAndTypeR, realisAndTypeF1 = 0.0, 0.0, 0.0
    bcub, ceafe, ceafm, muc, blanc, averageCoref = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    startMentionScoring = False
    startCoreferenceScoring = False
    for line in ous.split('\n'):
        line = line.strip()
        if line == '=======Final Mention Detection Results=========':
            startMentionScoring = True
            startCoreferenceScoring = False
            continue
        if line == '=======Final Mention Coreference Results=========':
            startMentionScoring = False
            startCoreferenceScoring = True
            continue
        if not startMentionScoring and not startCoreferenceScoring: continue
        if startMentionScoring and line.startswith('plain'):
            els = line.split('\t')
            spanP, spanR, spanF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if startMentionScoring and line.startswith('mention_type') and not line.startswith('mention_type+realis_status'):
            els = line.split('\t')
            subtypeP, subtypeR, subtypeF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if startMentionScoring and line.startswith('realis_status'):
            els = line.split('\t')
            realisP, realisR, realisF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if startMentionScoring and line.startswith('mention_type+realis_status'):
            els = line.split('\t')
            realisAndTypeP, realisAndTypeR, realisAndTypeF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if startCoreferenceScoring and 'bcub' in line:
            els = line.split()
            bcub = float(els[4])
            continue
        if startCoreferenceScoring and 'ceafe' in line:
            els = line.split()
            ceafe = float(els[4])
            continue
        if startCoreferenceScoring and 'ceafm' in line:
            els = line.split()
            ceafm = float(els[4])
            continue
        if startCoreferenceScoring and 'muc' in line:
            els = line.split()
            muc = float(els[4])
            continue
        if startCoreferenceScoring and 'blanc' in line:
            els = line.split()
            blanc = float(els[4])
            continue
        if startCoreferenceScoring and 'Overall Average CoNLL score' in line:
            els = line.split()
            averageCoref = float(els[4])
            continue
        
    
    return OrderedDict({'spanP' : spanP, 'spanR' : spanR, 'spanF1' : spanF1,
                        'typeP' : subtypeP, 'typeR' : subtypeR, 'typeF1' : subtypeF1,
                        'subtypeP' : subtypeP, 'subtypeR' : subtypeR, 'subtypeF1' : subtypeF1,
                        'realisP' : realisP, 'realisR' : realisR, 'realisF1' : realisF1,
                        'realisAndTypeP' : realisAndTypeP, 'realisAndTypeR' : realisAndTypeR, 'realisAndTypeF1' : realisAndTypeF1,
                        'bcub' : bcub, 'ceafe' : ceafe, 'ceafm' : ceafm, 'muc' : muc, 'blanc' : blanc, 'averageCoref' : averageCoref})

def generateParameterFileName(model, expected_features1, expected_features2, nhidden, conv_feature_map, conv_win_feature_map, multilayerNN1, multilayerNN2):
    res = 'realis.' + model + '.f1-'
    for fe in expected_features1: res += str(expected_features1[fe])
    res += '.f2-'
    for fe in expected_features2: res += str(expected_features2[fe])
    res += '.h-' + str(nhidden)
    res += '.cf-' + str(conv_feature_map)
    res += '.cwf-'
    for wi in conv_win_feature_map: res += str(wi)
    res += '.mul1-'
    for mu in multilayerNN1: res += str(mu)
    res += '.mul2-'
    for mu in multilayerNN2: res += str(mu)
    return res

def train(dataset_path='',
          eventTypePath='',
          model='basic',
          wedWindow=-1,
          window=31,
          expected_features1 = OrderedDict([('anchor', -1), ('pos', -1), ('chunk', -1), ('possibleTypes', -1), ('dep', -1), ('nonref', -1), ('title', -1), ('eligible', -1)]),
          expected_features2 = OrderedDict([('anchor', -1), ('pos', -1), ('chunk', -1), ('possibleTypes', -1), ('dep', -1), ('nonref', -1), ('title', -1), ('eligible', -1)]),
          givenPath=None,
          withEmbs=False, # using word embeddings to initialize the network or not
          updateEmbs=True,
          optimizer='adadelta',
          lr=0.01,
          dropout=0.05,
          regularizer=0.5,
          norm_lim = -1.0,
          verbose=1,
          decay=False,
          batch=50,
          binaryCutoff=1,
          coreferenceCutoff=1,
          multilayerNN1=[1200, 600],
          multilayerNN2=[1200, 600],
          nhidden=100,
          conv_feature_map=100,
          conv_win_feature_map=[2,3,4,5],
          seed=3435,
          #emb_dimension=300, # dimension of word embedding
          nepochs=50,
          folder='./res'):
          
    if binaryCutoff > 0 and not model.startswith('#'): model = '#' + model
    
    folder = '/scratch/thn235/projects/nugget/resCoref/' + folder

    paramFolder = folder + '/params'

    if not os.path.exists(folder): os.mkdir(folder)
    if not os.path.exists(paramFolder): os.mkdir(paramFolder)
    
    paramFileName = paramFolder + '/' + generateParameterFileName(model, expected_features1, expected_features2, nhidden, conv_feature_map, conv_win_feature_map, multilayerNN1, multilayerNN2)

    print 'loading dataset: ', dataset_path, ' ...'
    revs, embeddings, dictionaries = cPickle.load(open(dataset_path, 'rb'))
    
    idx2word  = dict((k,v) for v,k in dictionaries['word'].iteritems())

    if not withEmbs:
        wordEmbs = embeddings['randomWord']
    else:
        print 'using word embeddings to initialize the network ...'
        wordEmbs = embeddings['word']
    emb_dimension = wordEmbs.shape[1]
    
    del embeddings['word']
    del embeddings['randomWord']
    embeddings['word'] = wordEmbs
    
    if expected_features1['dep'] >= 0: expected_features1['dep'] = 1
    if expected_features1['possibleTypes'] >= 0: expected_features1['possibleTypes'] = 1
    if expected_features2['dep'] >= 0: expected_features2['dep'] = 1
    if expected_features2['possibleTypes'] >= 0: expected_features2['possibleTypes'] = 1

    features1 = OrderedDict([('word', 0)])
    features2 = OrderedDict([('word', 0)])

    for ffin in expected_features1:
        features1[ffin] = expected_features1[ffin]
        if expected_features1[ffin] == 0:
            print 'using features1: ', ffin, ' : embeddings'
        elif expected_features1[ffin] == 1:
            print 'using features1: ', ffin, ' : binary'
            
    for ffin in expected_features2:
        features2[ffin] = expected_features2[ffin]
        if expected_features2[ffin] == 0:
            print 'using features2: ', ffin, ' : embeddings'
        elif expected_features2[ffin] == 1:
            print 'using features2: ', ffin, ' : binary'
    
    #preparing transfer knowledge
    kGivens = {}
    if givenPath and os.path.exists(givenPath):
        print '****Loading given knowledge in: ', givenPath
        kGivens = cPickle.load(open(givenPath, 'rb'))
    else: print givenPath, ' not exist'
    
    if 'window' in kGivens:
        print '<------ Using given window for instances ------>: ', kGivens['window']
        window = kGivens['window']
    if window % 2 == 0: window += 1
    
    datasets, idMappings, wholeLineEventTypeRealisStorer = make_data(revs, dictionaries, embeddings, features1, features2, window, eventTypePath)
    
    dimCorpus = datasets['train']
    
    vocsize = len(idx2word)
    nclasses = 2
    nsentences = len(dimCorpus['word1'])

    print 'vocabsize = ', vocsize, ', nclasses = ', nclasses, ', nsentences = ', nsentences, ', word embeddings dim = ', emb_dimension
    
    features_dim1 = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features1:
        if features1[ffin] == 1:
            features_dim1[ffin] = len(dimCorpus[ffin + '1'][0][0])
        elif features1[ffin] == 0:
            features_dim1[ffin] = embeddings[ffin].shape[1]
            
    features_dim2 = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features2:
        if features2[ffin] == 1:
            features_dim2[ffin] = len(dimCorpus[ffin + '2'][0][0])
        elif features2[ffin] == 0:
            features_dim2[ffin] = embeddings[ffin].shape[1]
    
    conv_winre = len(dimCorpus['word1'][0])
    
    print '------- length of the instances: ', conv_winre
    #binaryFeatureDim = -1
    
    if 'binaryFeatureDict' in kGivens:
        print '********** USING BINARY FEATURE DICTIONARY FROM LOADED MODEL'
        binaryFeatureDict = kGivens['binaryFeatureDict']
    else:
        print '********** CREATING BINARY FEATURE DICTIONARY FROM TRAINING DATA'
        binaryFeatureDict = makeBinaryDictionary(dimCorpus, binaryCutoff)
    maxBinaryFetDim = findMaximumBinaryLength(datasets)
    binaryFeatureDim = convertBinaryFeatures(datasets, maxBinaryFetDim, binaryFeatureDict)
    
    if 'coreferenceFeatureDict' in kGivens:
        print '********** USING COREFERENCE FEATURE DICTIONARY FROM LOADED MODEL'
        coreferenceFeatureDict = kGivens['coreferenceFeatureDict']
    else:
        print '********** CREATING COREFERENCE FEATURE DICTIONARY FROM TRAINING DATA'
        coreferenceFeatureDict = makeCoreferenceFeatureDictionary(dimCorpus, coreferenceCutoff)
    coreferenceFeatureDim = convertCoreferenceFeatures(datasets, coreferenceFeatureDict)
    
    params = {'model' : model,
              'wedWindow' : wedWindow,
              'window' : window,
              'kGivens' : kGivens,
              'nh' : nhidden,
              'nc' : nclasses,
              'ne' : vocsize,
              'batch' : batch,
              'embs' : embeddings,
              'dropout' : dropout,
              'regularizer': regularizer,
              'norm_lim' : norm_lim,
              'updateEmbs' : updateEmbs,
              'features1' : features1,
              'features_dim1' : features_dim1,
              'features2' : features2,
              'features_dim2' : features_dim2,
              'optimizer' : optimizer,
              'binaryCutoff' : binaryCutoff,
              'binaryFeatureDim' : binaryFeatureDim,
              'binaryFeatureDict' : binaryFeatureDict,
              'coreferenceCutoff' : coreferenceCutoff,
              'coreferenceFeatureDim' : coreferenceFeatureDim,
              'coreferenceFeatureDict' : coreferenceFeatureDict,
              'multilayerNN1' : multilayerNN1,
              'multilayerNN2' : multilayerNN2,
              'conv_winre' : conv_winre,
              'conv_feature_map' : conv_feature_map,
              'conv_win_feature_map' : conv_win_feature_map}
    
    for corpus in datasets:
        for ed in datasets[corpus]:
            if ed == 'label' or ed == 'id':
                datasets[corpus][ed] = numpy.array(datasets[corpus][ed], dtype='int32')
            else:
                dty = 'float32' if numpy.array(datasets[corpus][ed][0]).ndim == 2 else 'int32'
                if ed == 'coreferenceFeatures': dtype = 'float32'
                datasets[corpus][ed] = numpy.array(datasets[corpus][ed], dtype=dty)
    
    trainCorpus = {}
    augt = datasets['train']
    if nsentences % batch > 0:
        extra_data_num = batch - nsentences % batch
        for ed in augt:
            numpy.random.seed(3435)
            permuted = numpy.random.permutation(augt[ed])   
            extra_data = permuted[:extra_data_num]
            trainCorpus[ed] = numpy.append(augt[ed],extra_data,axis=0)
    else:
        for ed in augt:
            trainCorpus[ed] = augt[ed]
    
    number_batch = trainCorpus['word1'].shape[0] / batch
    
    print '... number of batches: ', number_batch
    
    # instanciate the model
    print 'building model ...'
    numpy.random.seed(seed)
    random.seed(seed)
    if model.startswith('#'):
        model = model[1:]
        params['model'] = model
        reModel = eval('hybridModel')(params)
    else: reModel = eval('mainModel')(params)
    print 'done'
    
    evaluatingDataset = OrderedDict([#('train', datasets['train']),
                                     ('valid', datasets['valid']),
                                     ('test', datasets['test'])
                                     ])
    
    _predictions, _probs, _perfs = OrderedDict(), OrderedDict(), OrderedDict()
    
    # training model
    best_f1 = -numpy.inf
    clr = lr
    s = OrderedDict()
    for e in xrange(nepochs):
        s['_ce'] = e
        tic = time.time()
        #nsentences = 5
        print '-------------------training in epoch: ', e, ' -------------------------------------'
        # for i in xrange(nsentences):
        miniId = -1
        for minibatch_index in numpy.random.permutation(range(number_batch)):
            miniId += 1
            trainIn = OrderedDict()
            for ed in features1:
                if features1[ed] >= 0:
                    if ed + '1' not in trainCorpus:
                        print 'cannot find data in train for: ', ed + '1'
                        exit()
                    
                    trainIn[ed + '1'] = trainCorpus[ed + '1'][minibatch_index*batch:(minibatch_index+1)*batch]
            for ed in features2:
                if features2[ed] >= 0:
                    if ed + '2' not in trainCorpus:
                        print 'cannot find data in train for: ', ed + '2'
                        exit()
                    
                    trainIn[ed + '2'] = trainCorpus[ed + '2'][minibatch_index*batch:(minibatch_index+1)*batch]

            trainAnchor1 = trainCorpus['position1'][minibatch_index*batch:(minibatch_index+1)*batch]
            trainAnchor2 = trainCorpus['position2'][minibatch_index*batch:(minibatch_index+1)*batch]

            zippedData = [ trainIn[ed] for ed in trainIn ]

            zippedData += [trainAnchor1, trainAnchor2]
            
            if 'coreferenceFeatures' in trainCorpus:
                zippedData += [trainCorpus['coreferenceFeatures'][minibatch_index*batch:(minibatch_index+1)*batch]]
            
            if 'binaryFeatures1' in trainCorpus:
                zippedData += [trainCorpus['binaryFeatures1'][minibatch_index*batch:(minibatch_index+1)*batch]]
                zippedData += [trainCorpus['binaryFeatures2'][minibatch_index*batch:(minibatch_index+1)*batch]]

            zippedData += [trainCorpus['label'][minibatch_index*batch:(minibatch_index+1)*batch]]
            
            reModel.f_grad_shared(*zippedData)
            reModel.f_update_param(clr)
            
            for ed in reModel.container['embDict']:
                reModel.container['setZero'][ed](reModel.container['zeroVecs'][ed])
                
            if verbose:
                if miniId % 50 == 0:
                    print 'epoch %i >> %2.2f%%'%(e,(miniId+1)*100./number_batch),'completed in %.2f (sec) <<'%(time.time()-tic)
                    sys.stdout.flush()

        # evaluation // back into the real world : idx -> words
        print 'evaluating in epoch: ', e

        for elu in evaluatingDataset:
            _predictions[elu], _probs[elu] = predict(evaluatingDataset[elu], batch, reModel, idx2word, features1, features2)
            
            writeout(evaluatingDataset[elu], _predictions[elu], _probs[elu], idMappings[elu], wholeLineEventTypeRealisStorer[elu], folder + '/' + elu + '.coref.pred' + str(e))
            
            if goldFiles[elu] and tokenFiles[elu]:
                _perfs[elu] = myScore(goldFiles[elu], folder + '/' + elu + '.coref.pred' + str(e), tokenFiles[elu], conllTempFile)
        
        perPrint(_perfs)
        
        print 'saving parameters ...'
        reModel.save(paramFileName + str(e) + '.pkl')
        
        if _perfs['valid']['averageCoref'] > best_f1:
            #rnn.save(folder)
            best_f1 = _perfs['valid']['averageCoref']
            print '*************NEW BEST: epoch: ', e
            if verbose:
                perPrint(_perfs, len('Current Performance')*'-')

            for elu in _perfs:
                s[elu] = _perfs[elu]
            s['_be'] = e
            
            #subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            #subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        
        # learning rate decay if no improvement in 10 epochs
        if decay and abs(s['_be']-s['_ce']) >= 10: clr *= 0.5 
        if clr < 1e-5: break

    print '>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>'
    print 'BEST RESULT: epoch: ', s['_be']
    perPrint(s, len('Current Performance')*'-')
    print ' with the model in ', folder

def perPrint(perfs, mess='Current Performance'):
    print '------------------------------%s-----------------------------'%mess
    for elu in perfs:
        if elu.startswith('_'): continue
        print '----', elu
        print 'plain: ', str(perfs[elu]['spanP']) + '\t' + str(perfs[elu]['spanR']) + '\t' + str(perfs[elu]['spanF1'])
        print 'mention_type: ', str(perfs[elu]['typeP']) + '\t' + str(perfs[elu]['typeR']) + '\t' + str(perfs[elu]['typeF1'])
        print 'mention_subtype: ', str(perfs[elu]['subtypeP']) + '\t' + str(perfs[elu]['subtypeR']) + '\t' + str(perfs[elu]['subtypeF1'])
        print 'realis_status: ', str(perfs[elu]['realisP']) + '\t' + str(perfs[elu]['realisR']) + '\t' + str(perfs[elu]['realisF1'])
        print 'mention_type+realis_status: ', str(perfs[elu]['realisAndTypeP']) + '\t' + str(perfs[elu]['realisAndTypeR']) + '\t' + str(perfs[elu]['realisAndTypeF1'])
        print 'bcub: ', perfs[elu]['bcub']
        print 'ceafe: ', perfs[elu]['ceafe']
        print 'ceafm: ', perfs[elu]['ceafm']
        print 'muc: ', perfs[elu]['muc']
        print 'blanc: ', perfs[elu]['blanc']
        print 'averageCoref: ', perfs[elu]['averageCoref']
    
    print '------------------------------------------------------------------------------'

if __name__ == '__main__':
    pass
