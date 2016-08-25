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
from model import *

goldFiles = OrderedDict([('train', ''),
                         ('valid', ''),
                         ('test', '')
                        ])
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

def make_data(revs, dictionaries, embeddings, features, window, eventTypePath):

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
    
    eventTypeStorer, realisStorer, wholeLineEventTypeRealisStorer = readEventTypeFile(eventTypePath)
    
    res = {}
    idMappings = {}
    for datn in revs:
        res[datn] = defaultdict(list)
        idMappings[datn] = {}
        iid = -1
        for doc in revs[datn]:
            instanceId = -1
            for rev in revs[datn][doc]['instances']:
            
                instanceId += 1
                ikey = datn + ' ' + doc + ' ' + str(instanceId)
                
                if ikey not in eventTypeStorer: continue
                if (datn != 'test') and (rev['type'] == 0 or rev['subtype'] == 0 or rev['realis'] == -1):
                    print 'should be an event istance here, but label from original data does not show that!'
                    exit()
                
                fetType, fetSubtype = parseTypeSubType(eventTypeStorer[ikey], rev, datn, typeDict, subtypeDict)
            
                ists = generateDataInstance(rev, dictionaries, embeddings, features, window)
                
                for kk in ists: res[datn][kk] += [ists[kk]]
                
                stFeatures = ['EventType=' + fetType, 'EventSubType=' + fetSubtype]
                res[datn]['binaryFeatures'] += [rev['binaryFeatures'] + stFeatures]
                
                res[datn]['position'] += [rev['anchor']]
                
                if datn != 'test' and realisDict[rev['realis']] != realisStorer[ikey]:
                    print 'realis mismatched in make data: ', datn, realisDict[rev['realis']], ' vs ', realisStorer[ikey]
                    exit()
                res[datn]['label'] += [rev['realis'] if datn != 'test' else -1]
                
                
                iid += 1
                idMappings[datn][iid] = ikey
                res[datn]['id'] += [iid]
    
    return res, idMappings, wholeLineEventTypeRealisStorer
    
def parseTypeSubType(st, rev, datn, typeDict, subtypeDict):
    els = st.split('_')
    fetType, fetSubtype = els[0], els[1]
    
    cpType, cpSubType = typeDict[rev['type']], subtypeDict[rev['subtype']]
    
    if datn != 'test' and fetType != cpType:
        print 'type from realis file and original dataset not matched: ', datn, fetType, ' vs ', cpType
        exit()
    if datn != 'test' and fetSubtype != cpSubType:
        print 'subtype from realis file and original dataset not matched: ', datn, fetSubtype, ' vs ', cpSubType
        exit()
    
    return fetType, fetSubtype
    
def readEventTypeFile(eventTypePath):
    ccps = ['train', 'test', 'valid']
    eventTypeStorer = {}
    realisStorer = {}
    wholeLineEventTypeRealisStorer = {}
    print 'reading ids of instances with types for realis ...'
    for ccp in ccps:
        with open(eventTypePath + '/' + ccp + '.txt') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('@'): continue
            
                els = line.split('\t')
                doc = els[1]
            
                #span = els[3]
                st = els[5].replace('-', '').lower()
                
                reli = els[6]
                
                instanceId = els[-1]
                
                key = ccp + ' ' + doc + ' ' + instanceId
                
                eventTypeStorer[key] = st
                realisStorer[key] = reli
                wholeLineEventTypeRealisStorer[key] = line
            
    return eventTypeStorer, realisStorer, wholeLineEventTypeRealisStorer

def makeBinaryDictionary(dat, cutoff=1):
    if cutoff < 0: return None
    print '-------creating binary feature dictionary on the training data--------'
    
    bfdCounter = defaultdict(int)
    for rev in dat['binaryFeatures']:
        for fet in rev: bfdCounter[fet] += 1
    print 'binary feature cutoff: ', cutoff
    bfd = {}
    for fet in bfdCounter:
        if bfdCounter[fet] >= cutoff:
            if fet not in bfd: bfd[fet] = len(bfd)
    
    print 'size of dictionary: ', len(bfd)
    
    return bfd

def findMaximumBinaryLength(dats):
    
    maxBiLen = -1
    for corpus in dats:
        for rev in dats[corpus]['binaryFeatures']:
            if len(rev) > maxBiLen: maxBiLen = len(rev)
    print 'maximum number of binary features: ', maxBiLen
    
    return maxBiLen

def convertBinaryFeatures(dat, maxBiLen, bfd):
    if not bfd:
        for corpus in dat: del dat[corpus]['binaryFeatures']
        return -1
    print 'converting binary features to vectors ...'
    for corpus in dat:
        for i in range(len(dat[corpus]['word'])):
            dat[corpus]['binaryFeatures'][i] = getBinaryVector(dat[corpus]['binaryFeatures'][i], maxBiLen, bfd)
            
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

def predict(corpus, batch, reModel, idx2word, idx2label, features):
    evaluateCorpus = {}
    extra_data_num = -1
    nsen = corpus['word'].shape[0]
    if nsen % batch > 0:
        extra_data_num = batch - nsen % batch
        for ed in corpus:  
            extra_data = corpus[ed][:extra_data_num]
            evaluateCorpus[ed] = numpy.append(corpus[ed],extra_data,axis=0)
    else:
        for ed in corpus: 
            evaluateCorpus[ed] = corpus[ed]
        
    numBatch = evaluateCorpus['word'].shape[0] / batch
    predictions_corpus = numpy.array([], dtype='int32')
    probs_corpus = []
    for i in range(numBatch):
        zippedCorpus = [ evaluateCorpus[ed][i*batch:(i+1)*batch] for ed in features if features[ed] >= 0 ]
        zippedCorpus += [ evaluateCorpus['position'][i*batch:(i+1)*batch] ]
        
        if 'binaryFeatures' in evaluateCorpus:
            zippedCorpus += [ evaluateCorpus['binaryFeatures'][i*batch:(i+1)*batch] ]
        
        clas, probs = reModel.classify(*zippedCorpus)
        predictions_corpus = numpy.append(predictions_corpus, clas)
        probs_corpus.append(probs)
    
    probs_corpus = numpy.concatenate(probs_corpus, axis=0)
    
    if extra_data_num > 0:
        predictions_corpus = predictions_corpus[0:-extra_data_num]
        probs_corpus = probs_corpus[0:-extra_data_num]

    return predictions_corpus, probs_corpus

def writeout(corpus, predictions, probs, idMapping, wholeLineEventTypeRealisStorer, idx2label, ofile):

    holder = defaultdict(list)
    for id, pred, prob in zip(corpus['id'], predictions, probs):
    
        if id not in idMapping:
            print 'cannot find id : ', id , ' in mapping'
            exit()
        ikey = idMapping[id]
        docMap = ikey.split()[1]
        
        if ikey not in wholeLineEventTypeRealisStorer:
            print 'cannot find key: ', key, ' in wholeLineEventTypeRealisStorer. It should be there as we only consider detected event instances now.'
            exit()
        
        line = wholeLineEventTypeRealisStorer[ikey]
        
        if pred not in idx2label:
            print 'cannot find realis prediction: ', pred, ' in idx2label'
            exit()
        realis = idx2label[pred]
        realisConfidence = numpy.max(prob)
        
        els = line.split('\t')
        els[6] = realis
        els[9] = str(realisConfidence)
        
        docOut = els[1]
        if docOut != docMap:
            print 'document mismatch in writeout for realis: ', docOut, ' vs ', docMap
            exit()
        
        out = '\t'.join(els)
        
        holder[docOut] += [out]
    
    writer = open(ofile, 'w')
    for doc in holder:
        writer.write('#BeginOfDocument ' + doc + '\n')
        for em in holder[doc]:
            writer.write(em + '\n')
        for i, em in enumerate(holder[doc]):
            id = em.split('\t')[2]
            writer.write('@Coreference' + '\t' + 'C' + str(i) + '\t' + id + '\n')
        writer.write('#EndOfDocument' + '\n')
    writer.close()

def myScore(goldFile, systemFile):
    
    gType, gSubType, gRealis = readAnnotationFile(goldFile)
    sType, sSubType, sRealis = readAnnotationFile(systemFile)
    
    totalSpan = 0
    for doc in gType: totalSpan += len(gType[doc])
    predictedSpan, correctSpan = 0, 0
    for doc in sType:
        predictedSpan += len(sType[doc])
        for span in sType[doc]:
            if doc in gType and span in gType[doc]: correctSpan += 1
            
    spanP, spanR, spanF1 = getPerformance(totalSpan, predictedSpan, correctSpan)
    
    totalType = 0
    for doc in gType:
        for span in gType[doc]:
            totalType += len(gType[doc][span])
    predictedType, correctType = 0, 0
    for doc in sType:
        predictedType += len(sType[doc])
        for span in sType[doc]:
            itype = next(iter(sType[doc][span])) #sType[doc][span][0]
            if doc in gType and span in gType[doc] and itype in gType[doc][span]:
                correctType += 1
    typeP, typeR, typeF1 = getPerformance(totalType, predictedType, correctType)
    
    totalSubType = 0
    for doc in gSubType:
        for span in gSubType[doc]:
            totalSubType += len(gSubType[doc][span])
    predictedSubType, correctSubType = 0, 0
    for doc in sSubType:
        predictedSubType += len(sSubType[doc])
        for span in sSubType[doc]:
            isubtype = next(iter(sSubType[doc][span])) #sSubType[doc][span][0]
            if doc in gSubType and span in gSubType[doc] and isubtype in gSubType[doc][span]:
                correctSubType += 1
    subtypeP, subtypeR, subtypeF1 = getPerformance(totalSubType, predictedSubType, correctSubType)
    
    totalRealis = 0
    for doc in gRealis:
        for span in gRealis[doc]:
            totalRealis += len(gRealis[doc][span])
    predictedRealis, correctRealis = 0, 0
    for doc in sRealis:
        predictedRealis += len(sSubType[doc])
        for span in sRealis[doc]:
            irealis = next(iter(sRealis[doc][span])) #sRealis[doc][span][0]
            if doc not in sSubType or span not in sSubType[doc]: continue
            isubtype = next(iter(sSubType[doc][span])) #sSubType[doc][span][0]
            if doc in gRealis and span in gRealis[doc] and doc in gSubType and span in gSubType[doc] and isubtype in gSubType[doc][span] and irealis in gRealis[doc][span]:
                correctRealis += 1
    realisP, realisR, realisF1 = getPerformance(totalRealis, predictedRealis, correctRealis)
    
    
    return OrderedDict({'spanP' : spanP, 'spanR' : spanR, 'spanF1' : spanF1,
                        'typeP' : typeP, 'typeR' : typeR, 'typeF1' : typeF1,
                        'subtypeP' : subtypeP, 'subtypeR' : subtypeR, 'subtypeF1' : subtypeF1,
                        'realisP' : realisP, 'realisR' : realisR, 'realisF1' : realisF1})

def getPerformance(total, predicted, correct):
    p = 0.0 if predicted == 0 else 1.0 * correct / predicted
    r = 0.0 if total == 0 else 1.0 * correct / total
    f1 = 0.0 if (p + r) == 0 else (2*p*r) / (p+r)
    return p, r, f1

def readAnnotationFile(afile):
    typeRes = defaultdict(lambda : defaultdict(set))
    subtypeRes = defaultdict(lambda : defaultdict(set))
    realisRes = defaultdict(lambda : defaultdict(set))
    with open(afile) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('@'): continue
            
            els = line.split('\t')
            doc = els[1]
            
            span = els[3]
            st = els[5].replace('-', '').lower()
            etype = st[0:st.find('_')]
            esubtype = st[(st.find('_')+1):]
            
            realis = els[6]
            
            typeRes[doc][span].add(etype) # += [etype]
            subtypeRes[doc][span].add(st) # += [st]
            realisRes[doc][span].add(realis) # += [realis]
    return typeRes, subtypeRes, realisRes

def saving(corpus, predictions, probs, idx2word, idx2label, idMapping, address):
        
    def generateEvtSent(rid, sent, anchor, start, end, pred, idx2word, idx2label, idMapping):
        res = idMapping[rid] + '\t'
        res += str(start) + ',' + str(end) + '\t'
        for i, w in enumerate(sent):
            if w == 0: continue
            w = idx2word[w]
            if i == anchor:
                res += '<anchor>' + w + '</anchor>' + ' '
            else:
                res += w + ' '
        
        res = res.strip()
        res += '\t' + idx2label[pred]
        
        return res
    
    def generateProb(rid, pro, start, end, idx2label, idMapping):
        res = idMapping[rid] + '\t'
        res += str(start) + ',' + str(end) + '\t'
        for i in range(pro.shape[0]):
            res += idx2label[i] + ':' + str(pro[i]) + ' '
        res = res.strip()
        return res
    
    fout = open(address, 'w')
    fprobOut = open(address + '.prob', 'w')
    

    for rid, sent, anchor, start, end, pred, pro in zip(corpus['id'], corpus['word'], corpus['position'], corpus['wordStart'], corpus['wordEnd'], predictions, probs):
        fout.write(generateEvtSent(rid, sent, anchor, start, end, pred, idx2word, idx2label, idMapping) + '\n')
        fprobOut.write(generateProb(rid, pro, start, end, idx2label, idMapping) + '\n')
    
    fout.close()
    fprobOut.close()

def generateParameterFileName(model, expected_features, nhidden, conv_feature_map, conv_win_feature_map, multilayerNN1, multilayerNN2):
    res = 'realis.' + model + '.f-'
    for fe in expected_features: res += str(expected_features[fe])
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
          expected_features = OrderedDict([('anchor', -1), ('pos', -1), ('chunk', -1), ('possibleTypes', -1), ('dep', -1), ('nonref', -1), ('title', -1), ('eligible', -1)]),
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
    
    folder = './resRealis/' + folder

    paramFolder = folder + '/params'

    if not os.path.exists(folder): os.mkdir(folder)
    if not os.path.exists(paramFolder): os.mkdir(paramFolder)
    
    paramFileName = paramFolder + '/' + generateParameterFileName(model, expected_features, nhidden, conv_feature_map, conv_win_feature_map, multilayerNN1, multilayerNN2)

    print 'loading dataset: ', dataset_path, ' ...'
    revs, embeddings, dictionaries = cPickle.load(open(dataset_path, 'rb'))
    
    idx2label = dict((k,v) for v,k in dictionaries['realis'].iteritems())
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
    
    if expected_features['dep'] >= 0: expected_features['dep'] = 1
    if expected_features['possibleTypes'] >= 0: expected_features['possibleTypes'] = 1

    features = OrderedDict([('word', 0)])

    for ffin in expected_features:
        features[ffin] = expected_features[ffin]
        if expected_features[ffin] == 0:
            print 'using features: ', ffin, ' : embeddings'
        elif expected_features[ffin] == 1:
            print 'using features: ', ffin, ' : binary'
    
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
    
    datasets, idMappings, wholeLineEventTypeRealisStorer = make_data(revs, dictionaries, embeddings, features, window, eventTypePath)
    
    dimCorpus = datasets['train']
    
    vocsize = len(idx2word)
    nclasses = len(idx2label)
    nsentences = len(dimCorpus['word'])

    print 'vocabsize = ', vocsize, ', nclasses = ', nclasses, ', nsentences = ', nsentences, ', word embeddings dim = ', emb_dimension
    
    features_dim = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features:
        if features[ffin] == 1:
            features_dim[ffin] = len(dimCorpus[ffin][0][0])
        elif features[ffin] == 0:
            features_dim[ffin] = embeddings[ffin].shape[1]
    
    conv_winre = len(dimCorpus['word'][0])
    
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
              'features' : features,
              'features_dim' : features_dim,
              'optimizer' : optimizer,
              'binaryCutoff' : binaryCutoff,
              'binaryFeatureDim' : binaryFeatureDim,
              'binaryFeatureDict' : binaryFeatureDict,
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
    
    number_batch = trainCorpus['word'].shape[0] / batch
    
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
    
    evaluatingDataset = OrderedDict([('train', datasets['train']),
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
            for ed in features:
                if features[ed] >= 0:
                    if ed not in trainCorpus:
                        print 'cannot find data in train for: ', ed
                        exit()
                    
                    trainIn[ed] = trainCorpus[ed][minibatch_index*batch:(minibatch_index+1)*batch]

            trainAnchor = trainCorpus['position'][minibatch_index*batch:(minibatch_index+1)*batch]

            zippedData = [ trainIn[ed] for ed in trainIn ]

            zippedData += [trainAnchor]
            
            if 'binaryFeatures' in trainCorpus:
                zippedData += [trainCorpus['binaryFeatures'][minibatch_index*batch:(minibatch_index+1)*batch]]

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
            _predictions[elu], _probs[elu] = predict(evaluatingDataset[elu], batch, reModel, idx2word, idx2label, features)
            
            writeout(evaluatingDataset[elu], _predictions[elu], _probs[elu], idMappings[elu], wholeLineEventTypeRealisStorer, idx2label, folder + '/' + elu + '.realis.pred' + str(e))
            
            _perfs[elu] = myScore(goldFiles[elu], folder + '/' + elu + '.realis.pred' + str(e))
        
        perPrint(_perfs)
        
        print 'saving parameters ...'
        reModel.save(paramFileName + str(e) + '.pkl')
        
        #print 'saving output ...'
        #for elu in evaluatingDataset:
        #    saving(evaluatingDataset[elu], _predictions[elu], _probs[elu], idx2word, idx2label, idMappings[elu], folder + '/' + elu + str(e) + '.realis.fullPred')
        
        if _perfs['valid']['realisF1'] > best_f1:
            #rnn.save(folder)
            best_f1 = _perfs['valid']['realisF1']
            print '*************NEW BEST: epoch: ', e
            if verbose:
                perPrint(_perfs, len('Current Performance')*'-')

            for elu in evaluatingDataset:
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
        print str(perfs[elu]['spanP']) + '\t' + str(perfs[elu]['spanR']) + '\t' + str(perfs[elu]['spanF1'])
        print str(perfs[elu]['typeP']) + '\t' + str(perfs[elu]['typeR']) + '\t' + str(perfs[elu]['typeF1'])
        print str(perfs[elu]['subtypeP']) + '\t' + str(perfs[elu]['subtypeR']) + '\t' + str(perfs[elu]['subtypeF1'])
        print str(perfs[elu]['realisP']) + '\t' + str(perfs[elu]['realisR']) + '\t' + str(perfs[elu]['realisF1'])
    
    print '------------------------------------------------------------------------------'

if __name__ == '__main__':
    pass
