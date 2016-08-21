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

#dataset_path = '/home/thn235/projects/extension/ver1/word2vec_transfer.pkl'
dataset_path = '/scratch/thn235/projects/extension/ver1/word2vec_transfer.pkl'

##################################################################
    
def generateDataInstance(rev, dictionaries, embeddings, features, mLen, task):

    numPosition = embeddings['dist1'].shape[0]-1
    numPartOfSpeech = embeddings['partOfSpeech'].shape[0]-1
    numChunk = embeddings['chunk'].shape[0]-1
    esuffix = 'Relation' if task == 'relation' else 'Event'
    numEntity = embeddings['entity' + esuffix].shape[0]-1
    numDependency = embeddings['dependency'].shape[0]-1

    x = []
    partOfSpeech = []
    chunk = []
    entity = []
    dependency = []
    dist1 = []
    if 'pos2' in rev: dist2 = []
    
    id = -1
    for word, paro, chu, ent, deps in zip(rev["text"], rev["partOfSpeech"], rev["chunk"], rev["entity"], rev["dependency"]):
        id += 1
        word = ' '.join(word.split('_'))
        if word in dictionaries["word"]:
        
            x.append(dictionaries["word"][word])
            
            depFet = [0] * numDependency
            for did in deps:
                depFet[did-1] = 1
            dependency.append(depFet)
            
            posFet = [0] * numPartOfSpeech
            posFet[paro-1] = 1
            partOfSpeech.append((posFet if features['partOfSpeech'] == 1 else paro))
            
            chuFet = [0] * numChunk
            chuFet[chu-1] = 1
            chunk.append((chuFet if features['chunk'] == 1 else chu))
            
            entFet = [0] * numEntity
            entFet[ent-1] = 1
            entity.append((entFet if features['entity'] == 1 else ent))
            
            #######pos
            lpos1 = numPosition / 2 + id - rev["pos1"]
            scalar_dist1 = (lpos1+1)
            vector_dist1 = [0] * numPosition
            vector_dist1[lpos1] = 1
            dist1.append((vector_dist1 if features['dist1'] == 1 else scalar_dist1))
            
            if 'pos2' in rev:
                lpos2 = numPosition / 2 + id - rev["pos2"]
                scalar_dist2 = (lpos2+1)
                vector_dist2 = [0] * numPosition
                vector_dist2[lpos2] = 1
                dist2.append((vector_dist2 if features['dist2'] == 1 else scalar_dist2))
                
        else:
            print 'unrecognized word'
            exit()
    
    if len(x) > mLen:
        print 'incorrect length!'
        exit()
    
    if len(x) < mLen:
        depFet = [0] * numDependency
        posFet = [0] * numPartOfSpeech
        chuFet = [0] * numChunk
        entFet = [0] * numEntity
        vector_dist1 = [0] * numPosition
        if 'pos2' in rev:
            vector_dist2 = [0] * numPosition
        while len(x) < mLen:
            x.append(0)
            dependency.append(depFet)
            partOfSpeech.append((posFet if features['partOfSpeech'] == 1 else 0))
            chunk.append((chuFet if features['chunk'] == 1 else 0))
            entity.append((entFet if features['entity'] == 1 else 0))
            dist1.append((vector_dist1 if features['dist1'] == 1 else 0))
            if 'pos2' in rev:
                dist2.append((vector_dist2 if features['dist2'] == 1 else 0))
    
    ret = {'word' : x, 'dist1' : dist1, 'partOfSpeech' : partOfSpeech, 'chunk' : chunk, 'entity' : entity, 'dependency' : dependency}
    if 'pos2' in rev: ret['dist2'] = dist2
    
    return ret

def make_data(revs, task, dictionaries, embeddings, features, tmode):

    mLen = -1
    for datn in revs:
        for rev in revs[datn]:
            if len(rev["text"]) > mLen:
                mLen = len(rev["text"])
    
    print 'maximum of length in the dataset: ', mLen
    
    res = {}
    for rev in revs[task]:
    
        sot = rev["sid"]
        sot = sot[0:sot.find('-')]
        if tmode == 'target' and sot == 'sp': continue
        if tmode == 'source' and sot == 'tp': continue
        
        ists = generateDataInstance(rev, dictionaries, embeddings, features, mLen, task)
         
        if rev["corpus"] not in res: res[rev["corpus"]] = defaultdict(list)
        
        for kk in ists:
            res[rev["corpus"]][kk] += [ists[kk]]
        
        res[rev["corpus"]]['label'] += [ rev["y"] ]
        res[rev["corpus"]]['pos1'] += [rev["pos1"]]
        if 'pos2' in rev:
            res[rev["corpus"]]['pos2'] += [rev["pos2"]]
        res[rev["corpus"]]['id'] += [rev["id"]]
        res[rev["corpus"]]['binaryFeatures'] += [rev["binaryFeatures"]]
    
    return res

def makeBinaryDictionary(dat, dats, cutoff=1):
    if cutoff < 0: return None, None
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
    
    maxBiLen = -1
    for corpus in dats:
        for rev in dats[corpus]['binaryFeatures']:
            if len(rev) > maxBiLen: maxBiLen = len(rev)
    print 'maximum number of binary features: ', maxBiLen
    
    return maxBiLen, bfd

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

def predict(corpus, batch, reModel, idx2word, idx2label, features, task):
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
        zippedCorpus += [ evaluateCorpus['pos1'][i*batch:(i+1)*batch] ]
        if task == 'relation':
            zippedCorpus += [ evaluateCorpus['pos2'][i*batch:(i+1)*batch] ]
        
        if 'binaryFeatures' in evaluateCorpus:
            zippedCorpus += [ evaluateCorpus['binaryFeatures'][i*batch:(i+1)*batch] ]
        
        clas, probs = reModel.classify(*zippedCorpus)
        predictions_corpus = numpy.append(predictions_corpus, clas)
        probs_corpus.append(probs)
    
    probs_corpus = numpy.concatenate(probs_corpus, axis=0)
    
    if extra_data_num > 0:
        predictions_corpus = predictions_corpus[0:-extra_data_num]
        probs_corpus = probs_corpus[0:-extra_data_num]
    
    groundtruth_corpus = corpus['label']
    
    if predictions_corpus.shape[0] != groundtruth_corpus.shape[0]:
        print 'length not matched!'
        exit()
    #words_corpus = [ map(lambda x: idx2word[x], w) for w in corpus['word']]

    #return predictions_corpus, groundtruth_corpus, words_corpus
    return predictions_corpus, probs_corpus, groundtruth_corpus

def score(predictions, groundtruths):

    zeros = numpy.zeros(predictions.shape, dtype='int')
    numPred = numpy.sum(numpy.not_equal(predictions, zeros))
    numKey = numpy.sum(numpy.not_equal(groundtruths, zeros))
    
    predictedIds = numpy.nonzero(predictions)
    preds_eval = predictions[predictedIds]
    keys_eval = groundtruths[predictedIds]
    correct = numpy.sum(numpy.equal(preds_eval, keys_eval))
    
    #numPred, numKey, correct = 0, 0, 0
    
    precision = 100.0 * correct / numPred if numPred > 0 else 0.0
    recall = 100.0 * correct / numKey if numKey > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0. else 0.0
    
    return {'p' : precision, 'r' : recall, 'f1' : f1}

def saving(corpus, predictions, probs, groundtruths, idx2word, idx2label, idx2type, address, task):
    
    def determineType(type, pos1, idx2type):
        type1 = type[pos1]
        if type.ndim == 2:
            nty1 = -1
            for i, v in enumerate(type1):
                if v == 1:
                    nty1 = i + 1
                    break
            if nty1 < 0:
                print 'negative type index'
                exit()
            type1 = nty1
        return idx2type[type1]
    
    def generateRelSent(rid, sent, pos1, pos2, type1, type2, pred, gold, idx2word, idx2label):
        res = str(rid) + '\t'
        for i, w in enumerate(sent):
            if w == 0: continue
            w = idx2word[w]
            #w = '_'.join(w.split())
            if i == pos1:
                res += '<ent1-type=' + type1 + '>' + w + '</ent1>' + ' '
            elif i == pos2:
                res += '<ent2-type=' + type2 + '>' + w + '</ent2>' + ' '
            else:
                res += w + ' '
        
        res = res.strip()
        res += '\t' + idx2label[gold] + '\t' + idx2label[pred] + '\t' + ('__TRUE_' if pred == gold else '__FALSE_')
        
        return res
        
    def generateEvtSent(rid, sent, pos, pred, gold, idx2word, idx2label):
        res = str(rid) + '\t'
        for i, w in enumerate(sent):
            if w == 0: continue
            w = idx2word[w]
            #w = '_'.join(w.split())
            if i == pos:
                res += '<anchor>' + w + '</anchor>' + ' '
            else:
                res += w + ' '
        
        res = res.strip()
        res += '\t' + idx2label[gold] + '\t' + idx2label[pred] + '\t' + ('__TRUE_' if pred == gold else '__FALSE_')
        
        return res
    
    def generateProb(rid, pro, gold, idx2label):
        res = str(rid) + '\t'
        for i in range(pro.shape[0]):
            res += idx2label[i] + ':' + str(pro[i]) + ' '
        res = res.strip() + '\t' + idx2label[gold]
        return res
    
    fout = open(address, 'w')
    fprobOut = open(address + '.prob', 'w')
    
    if task == 'relation':
        for rid, sent, pos1, pos2, type, pred, pro, gold in zip(corpus['id'], corpus['word'], corpus['pos1'], corpus['pos2'], corpus['entity'], predictions, probs, groundtruths):
            type1 = determineType(type, pos1, idx2type)
            type2 = determineType(type, pos2, idx2type)
            fout.write(generateRelSent(rid, sent, pos1, pos2, type1, type2, pred, gold, idx2word, idx2label) + '\n')
            fprobOut.write(generateProb(rid, pro, gold, idx2label) + '\n')
    else:
        for rid, sent, pos1, pred, pro, gold in zip(corpus['id'], corpus['word'], corpus['pos1'], predictions, probs, groundtruths):
            fout.write(generateEvtSent(rid, sent, pos1, pred, gold, idx2word, idx2label) + '\n')
            fprobOut.write(generateProb(rid, pro, gold, idx2label) + '\n')
    
    fout.close()
    fprobOut.close()

def generateParameterFileName(task, model, expected_features, nhidden, conv_feature_map, conv_win_feature_map, multilayerNN1):
    res = task + '.' + model + '.f-'
    for fe in expected_features: res += str(expected_features[fe])
    res += '.h-' + str(nhidden)
    res += '.cf-' + str(conv_feature_map)
    res += '.cwf-'
    for wi in conv_win_feature_map: res += str(wi)
    res += '.mul-'
    for mu in multilayerNN1: res += str(mu)
    res += '.pkl'
    return res

def isWeightConv(conv_win_feature_map, kn):
    for i, conWin in enumerate(conv_win_feature_map):
        if kn.endswith('_win' + str(i) + '_conv_W_' + str(conWin)) and not kn.startswith('_ab_'): return True
    return False

def train(model='basic',
          #encoding='ffBiDirect',
          task='relation',
          tmode='union',
          wedWindow=-1,
          expected_features = OrderedDict([('partOfSpeech', -1), ('chunk', -1), ('entity', -1), ('dependency', -1), ('dist1', -1), ('dist2', -1)]),
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
    
    if task != 'event':
        print 'unrecognized task: ', task
        exit()
    
    #folder = '/home/thn235/projects/extension/ver1/res/' + folder
    folder = '/scratch/thn235/projects/extension/ver1/res/' + folder

    paramFolder = folder + '/params'

    if not os.path.exists(folder): os.mkdir(folder)
    if not os.path.exists(paramFolder): os.mkdir(paramFolder)
    
    paramFileName = paramFolder + '/' + generateParameterFileName(task, model, expected_features, nhidden, conv_feature_map, conv_win_feature_map, multilayerNN1)

    print 'loading dataset: ', dataset_path, ' ...'
    revs, embeddings, dictionaries = cPickle.load(open(dataset_path, 'rb'))
    
    idx2label = dict((k,v) for v,k in dictionaries[task + 'Label'].iteritems())
    idx2word  = dict((k,v) for v,k in dictionaries['word'].iteritems())
    esuffix = 'Relation' if task == 'relation' else 'Event'
    idx2type = dict((k,v) for v,k in dictionaries['entity' + esuffix].iteritems())

    if not withEmbs:
        wordEmbs = embeddings['randomWord']
    else:
        print 'using word embeddings to initialize the network ...'
        wordEmbs = embeddings['word']
    emb_dimension = wordEmbs.shape[1]
    
    embs = {'word' : wordEmbs,
            'dist1' : embeddings['dist1'],
            #'dist2' : embeddings['dist2'],
            'partOfSpeech' : embeddings['partOfSpeech'],
            'chunk' : embeddings['chunk'],
            'entity' : embeddings['entity' + esuffix],
            'dependency' : embeddings['dependency']}
    
    if task == 'event': del expected_features['dist2'] # = -1
    if expected_features['dependency'] >= 0: expected_features['dependency'] = 1

    features = OrderedDict([('word', 0)])

    for ffin in expected_features:
        features[ffin] = expected_features[ffin]
        if expected_features[ffin] == 0:
            print 'using features: ', ffin, ' : embeddings'
        elif expected_features[ffin] == 1:
            print 'using features: ', ffin, ' : binary'
        
    datasets = make_data(revs, task, dictionaries, embeddings, features, tmode)
    
    dimCorpus = datasets['train']
    
    maxBinaryFetDim, binaryFeatureDict = makeBinaryDictionary(dimCorpus, datasets, binaryCutoff)
    binaryFeatureDim = convertBinaryFeatures(datasets, maxBinaryFetDim, binaryFeatureDict)
    
    vocsize = len(idx2word)
    nclasses = len(idx2label)
    nsentences = len(dimCorpus['word'])

    print 'vocabsize = ', vocsize, ', nclasses = ', nclasses, ', nsentences = ', nsentences, ', word embeddings dim = ', emb_dimension
    
    features_dim = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features:
        features_dim[ffin] = ( len(dimCorpus[ffin][0][0]) if (features[ffin] == 1) else embs[ffin].shape[1] )
    
    conv_winre = len(dimCorpus['word'][0])
    
    print '------- length of the instances: ', conv_winre
    #binaryFeatureDim = -1
    
    #preparing transfer knowledge
    kGivens = {}
    if givenPath and os.path.exists(givenPath):
        print '****Loading given knowledge in: ', givenPath
        kGivens = cPickle.load(open(givenPath, 'rb'))
    else: print givenPath, ' not exist'
    #entStart = 0
    #for fedi in features:
    #    if fedi == 'entity': continue
    #    if features[fedi] >= 0: entStart += features_dim[fedi]
    #entEnd = entStart
    #if features['entity'] >= 0: entEnd += features_dim['entity']
    #if givenPath and os.path.exists(givenPath):
    #    print '****Loading given knowledge in: ', givenPath
    #    kGivens = cPickle.load(open(givenPath, 'rb'))
    #    del kGivens['entity']
    #    if task == 'relation':
    #        if 'dist1' not in kGivens or 'dist2' in kGivens:
    #            print 'cannot find dist1 or dist2 already in kGivens'
    #            exit()
    #        kGivens['dist2'] = numpy.copy(kGivens['dist1'])
    #dimDist = features_dim['dist1']
    #for kn in kGivens:
    #    if isWeightConv(conv_win_feature_map, kn):
    #        if entEnd > entStart:
    #            kGivens[kn][:, :, :, entStart:entEnd] = numpy.random.uniform(-0.25,0.25,(kGivens[kn].shape[0], kGivens[kn].shape[1], kGivens[kn].shape[2], entEnd-entStart))
    #        if task == 'event' and features['dist1'] >= 0:
    #            kGivens[kn] = kGivens[kn][:,:,:,0:-dimDist]
    #        if task == 'relation' and features['dist1'] >= 0:
    #            kGivens[kn] = numpy.append(kGivens[kn], kGivens[kn][:,:,:,-dimDist:], axis=3)
        
    #    if (kn.endswith('_Wx') or kn.endswith('_Wc')) and not kn.startswith('_ab_'):
    #        if entEnd > entStart:
    #            kGivens[kn][entStart:entEnd,:] = numpy.random.uniform(-0.25,0.25,(entEnd-entStart, kGivens[kn].shape[1]))
    #        if task == 'event' and features['dist1'] >= 0:
    #            kGivens[kn] = kGivens[kn][0:-dimDist,:]
    #        if task == 'relation' and features['dist1'] >= 0:
    #            kGivens[kn] = numpy.append(kGivens[kn], kGivens[kn][-dimDist:,:], axis=0)
    ##
    
    params = {#'encoding' : encoding,
              'model' : model,
              'task' : task,
              'wedWindow' : wedWindow,
              'kGivens' : kGivens,
              'nh' : nhidden,
              'nc' : nclasses,
              'ne' : vocsize,
              'batch' : batch,
              'embs' : embs,
              'dropout' : dropout,
              'regularizer': regularizer,
              'norm_lim' : norm_lim,
              'updateEmbs' : updateEmbs,
              'features' : features,
              'features_dim' : features_dim,
              'optimizer' : optimizer,
              'binaryCutoff' : binaryCutoff,
              'binaryFeatureDim' : binaryFeatureDim,
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
    
    trainCorpus = {} #evaluatingDataset['train']
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
    
    _predictions, _probs, _groundtruth, _perfs = OrderedDict(), OrderedDict(), OrderedDict(), OrderedDict() #, _words
    
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

            trainPos1 = trainCorpus['pos1'][minibatch_index*batch:(minibatch_index+1)*batch]

            zippedData = [ trainIn[ed] for ed in trainIn ]

            zippedData += [trainPos1]
            
            if task == 'relation':
                trainPos2 = trainCorpus['pos2'][minibatch_index*batch:(minibatch_index+1)*batch]
                zippedData += [trainPos2]
            
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
            _predictions[elu], _probs[elu], _groundtruth[elu] = predict(evaluatingDataset[elu], batch, reModel, idx2word, idx2label, features, task)
            _perfs[elu] = score(_predictions[elu], _groundtruth[elu])# folder + '/' + elu + '.txt'

        # evaluation // compute the accuracy using conlleval.pl

        #res_train = {'f1':'Not for now', 'p':'Not for now', 'r':'Not for now'}
        perPrint(_perfs)
        
        if _perfs['valid']['f1'] > best_f1:
            #rnn.save(folder)
            best_f1 = _perfs['valid']['f1']
            print '*************NEW BEST: epoch: ', e
            if verbose:
                perPrint(_perfs, len('Current Performance')*'-')

            for elu in evaluatingDataset:
                s[elu] = _perfs[elu]
            s['_be'] = e
            
            print 'saving parameters ...'
            reModel.save(paramFileName)
            print 'saving output ...'
            for elu in evaluatingDataset:
                saving(evaluatingDataset[elu], _predictions[elu], _probs[elu], _groundtruth[elu], idx2word, idx2label, idx2type, folder + '/' + elu + '.best.txt', task)
            #subprocess.call(['mv', folder + '/current.test.txt', folder + '/best.test.txt'])
            #subprocess.call(['mv', folder + '/current.valid.txt', folder + '/best.valid.txt'])
        else:
            print ''
        
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
        if elu.startswith('_'):
            continue
        pri = elu + ' : ' + str(perfs[elu]['p']) + '\t' + str(perfs[elu]['r'])+ '\t' + str(perfs[elu]['f1'])
        print pri
    
    print '------------------------------------------------------------------------------'
    
if __name__ == '__main__':
    pass
