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
                         ('valid', '/scratch/thn235/projects/nugget/golds/nugget/eval.tbf'),
                         ('test', '')
                        ])
tokenFiles = OrderedDict([('train', ''),
                          ('valid', '/scratch/thn235/projects/nugget/golds/nugget/tkn/'),
                          ('test', '')
                         ])

scoreScript = '/scratch/thn235/projects/nugget/scorer/scorer_v1.7.py'
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
                   
subtype2fullType = {"declarebankruptcy": "Business_Declare-Bankruptcy",

                    "artifact": "Manufacture_Artifact",
                   
                    "startposition": "Personnel_Start-Position",
                    "endposition": "Personnel_End-Position",
                    "nominate": "Personnel_Nominate",
                    "elect": "Personnel_Elect",
                   
                    "demonstrate": "Conflict_Demonstrate",
                    "attack": "Conflict_Attack",
                   
                    "broadcast": "Contact_Broadcast",
                    "contact": "Contact_Contact",
                    "correspondence": "Contact_Correspondence",
                    "meet": "Contact_Meet",
                   
                    "transfermoney": "Transaction_Transfer-Money",
                    "transferownership": "Transaction_Transfer-Ownership",
                    "transaction": "Transaction_Transaction",
                   
                    "transportartifact": "Movement_Transport-Artifact",
                    "transportperson": "Movement_Transport-Person",
                   
                    "startorg": "Business_Start-Org",
                    "endorg": "Business_End-Org",
                    "mergeorg": "Business_Merge-Org",
                   
                    "die": "Life_Die",
                    "divorce": "Life_Divorce",
                    "marry": "Life_Marry",
                    "beborn": "Life_Beborn",
                    "injure": "Life_Injure",
                   
                    "pardon": "Justice_Pardon",
                    "sue": "Justice_Sue",
                    "convict": "Justice_Convict",
                    "chargeindict": "Justice_Charge-Indict",
                    "trialhearing": "Justice_Trial-Hearing",
                    "sentence": "Justice_Sentence",
                    "appeal": "Justice_Appeal",
                    "releaseparole": "Justice_Release-Parole",
                    "extradite": "Justice_Extradite",
                    "fine": "Justice_Fine",
                    "execute": "Justice_Execute",
                    "arrestjail": "Justice_Arrest-Jail",
                    "acquit": "Justice_Acquit"}
                    
bothDieAndAttack = ['murder', 'murdered', 'murdering', 'assassinate', 'assassinated', 'assassinating', 'assassination', 'kill', 'killed', 'killing']

##################################################################

def generateDataInstance(rev, dictionaries, embeddings, features, binaryLenMap, window):
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
    
    res = defaultdict(list)
    for id in range(len(rev['word'])):
        for fet in features:
            if fet == 'word':
                _fword = rev['word'][id].lower()
                #if '-election' in _fword or '-elected' in _fword: _fword = _fword[(_fword.find('-')+1):]
                #if 'election-' in _fword: _fword = 'election'
                if _fword not in dictionaries['word']:
                    print 'cannot find id for word: ', _fword
                    exit()
                res['word'] += [ dictionaries['word'][_fword] ]
                continue
            
            if fet == 'anchor':
                anchor = numAnchor / 2 + id - rev['anchor']
                scalar_anchor = anchor+1
                vector_anchor = [-1] * binaryLenMap['anchor']
                vector_anchor[0] = anchor
                res['anchor'].append(vector_anchor if features['anchor'] == 1 else scalar_anchor)
                continue
            
            if fet == 'possibleTypes' or fet == 'dep':
                vector_fet = [-1] * binaryLenMap[fet]
                pdflen = len(rev[fet][id])
                vector_fet[0:pdflen] = rev[fet][id]
                res[fet].append(vector_fet)
                continue
            
            scalar_fet = rev[fet][id]
            vector_fet = [-1] * binaryLenMap[fet]
            if scalar_fet > 0:
                vector_fet[0] = scalar_fet-1
            res[fet].append(vector_fet if features[fet] == 1 else scalar_fet)
    
    return res

def make_data(revs, dictionaries, embeddings, features, window):

    mLen, mLenPossibleType, mLenDep = -1, -1, -1
    for datn in revs:
        for doc in revs[datn]:
            for ins in revs[datn][doc]['instances']:
                if len(ins['word']) > mLen:
                    mLen = len(ins['word'])
                
                for id in range(len(ins['word'])):
                    if len(ins['possibleTypes'][id]) > mLenPossibleType: mLenPossibleType = len(ins['possibleTypes'][id])
                    if len(ins['dep'][id]) > mLenDep: mLenDep = len(ins['dep'][id]) 
    
    print 'maximum length in the original dataset: ', mLen
    print 'maximum possibleType length in the original dataset: ', mLenPossibleType
    print 'maximum dep length in the original dataset: ', mLenDep
    print 'using window for instances: ', window

    binaryDimMap, binaryLenMap = {}, {}
    for fet in features:
        if fet == 'word': continue
        
        if fet == 'anchor': dim = embeddings['anchor'].shape[0]-1
        elif fet == 'possibleTypes' or fet == 'dep': dim = len(dictionaries[fet])
        else: dim = len(dictionaries[fet])-1
        binaryDimMap[fet] = dim
        
        bmlen = 1
        if fet == 'possibleTypes': bmlen = mLenPossibleType
        if fet == 'dep': bmlen = mLenDep
        binaryLenMap[fet] = bmlen
    
    res = {}
    idMappings = {}
    for datn in revs:
        res[datn] = defaultdict(list)
        idMappings[datn] = {}
        iid = -1
        for doc in revs[datn]:
            instanceId = -1
            for rev in revs[datn][doc]['instances']:
                ists = generateDataInstance(rev, dictionaries, embeddings, features, binaryLenMap, window)
                
                for kk in ists: res[datn][kk] += [ists[kk]]
                
                res[datn]['binaryFeatures'] += [rev['binaryFeatures']]
                res[datn]['label'] += [rev['subtype']]
                res[datn]['position'] += [rev['anchor']]
                
                res[datn]['wordStart'] += [rev['wordStart']]
                res[datn]['wordEnd'] += [rev['wordEnd']]
                
                iid += 1
                instanceId += 1
                ikey = datn + ' ' + doc + ' ' + str(instanceId)
                idMappings[datn][iid] = ikey
                res[datn]['id'] += [iid]
    
    return res, idMappings, binaryDimMap, binaryLenMap

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

def predict(corpus, batch, reModel, idx2word, idx2label, features, binTokenFets, binTokenIndexes):
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
        zippedCorpus = []
        for ed in features:
            if features[ed] < 0: continue
            fetbat = evaluateCorpus[ed][i*batch:(i+1)*batch]
            if features[ed] == 1:
                fetbat = produceBinaryTokenFeatures(fetbat, binTokenFets[ed], binTokenIndexes[ed])
            zippedCorpus += [ fetbat ]
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
    
#def writeout(corpus, predictions, probs, revs, idMapping, idx2word, idx2label, ofile):

#    counter = -1
#    holder = defaultdict(list)
#    for id, pred, prob in zip(corpus['id'], predictions, probs):
    
#        if pred == 0: continue
    
#        counter += 1
#        if id not in idMapping:
#            print 'cannot find id : ', id , ' in mapping'
#            exit()
#        ikey = idMapping[id]
#        keyls = ikey.split()
#        doc = keyls[1]
#        instanceId = int(keyls[2])
        
#        start = revs[doc]['instances'][instanceId]['wordStart']
#        end = revs[doc]['instances'][instanceId]['wordEnd']
#        anchor = revs[doc]['instances'][instanceId]['anchor']
#        word = revs[doc]['instances'][instanceId]['word'][anchor]
#        if pred not in idx2label:
#            print 'cannot find prediction: ', pred, ' in idx2label'
#            exit()
#        subtype = idx2label[pred]
#        if subtype not in subtype2typeMap:
#            print 'cannot find subtype: ', subtype, ' in mapping'
#            exit()
#        type = subtype2typeMap[subtype]
#        subTypeConfidence = numpy.max(prob)
        
#        out = 'NYU'
#        out += '\t' + doc
#        out += '\t' + 'E' + str(counter)
#        out += '\t' + str(start) + ',' + str(end)
#        out += '\t' + word
#        out += '\t' + type + '_' + subtype
#        out += '\t' + 'NONE'
#        out += '\t' + '1.0'
#        out += '\t' + str(subTypeConfidence)
#        out += '\t' + '1.0'
        
#        out += '\t' + str(instanceId)
        
#        holder[doc] += [out]
    
#    writer = open(ofile, 'w')
#    for doc in holder:
#        writer.write('#BeginOfDocument ' + doc + '\n')
#        for em in holder[doc]:
#            writer.write(em + '\n')
#        for i, em in enumerate(holder[doc]):
#            id = em.split('\t')[2]
#            writer.write('@Coreference' + '\t' + 'C' + str(i) + '\t' + id + '\n')
#        writer.write('#EndOfDocument' + '\n')
#    writer.close()

#def myScore(goldFile, systemFile):
    
#    gType, gSubType = readAnnotationFile(goldFile)
#    sType, sSubType = readAnnotationFile(systemFile)
    
#    totalSpan = 0
#    for doc in gType: totalSpan += len(gType[doc])
#    predictedSpan, correctSpan = 0, 0
#    for doc in sType:
#        predictedSpan += len(sType[doc])
#        for span in sType[doc]:
#            if doc in gType and span in gType[doc]: correctSpan += 1
            
#    spanP, spanR, spanF1 = getPerformance(totalSpan, predictedSpan, correctSpan)
    
#    totalType = 0
#    for doc in gType:
#        for span in gType[doc]:
#            totalType += len(gType[doc][span])
#    predictedType, correctType = 0, 0
#    for doc in sType:
#        predictedType += len(sType[doc])
#        for span in sType[doc]:
#            itype = next(iter(sType[doc][span])) #sType[doc][span][0]
#            if doc in gType and span in gType[doc] and itype in gType[doc][span]:
#                correctType += 1
#    typeP, typeR, typeF1 = getPerformance(totalType, predictedType, correctType)
    
#    totalSubType = 0
#    for doc in gSubType:
#        for span in gSubType[doc]:
#            totalSubType += len(gSubType[doc][span])
#    predictedSubType, correctSubType = 0, 0
#    for doc in sSubType:
#        predictedSubType += len(sSubType[doc])
#        for span in sSubType[doc]:
#            isubtype = next(iter(sSubType[doc][span])) #sSubType[doc][span][0]
#            if doc in gSubType and span in gSubType[doc] and isubtype in gSubType[doc][span]:
#                correctSubType += 1
#    subtypeP, subtypeR, subtypeF1 = getPerformance(totalSubType, predictedSubType, correctSubType)
    
#    return OrderedDict({'spanP' : spanP, 'spanR' : spanR, 'spanF1' : spanF1,
#                        'typeP' : typeP, 'typeR' : typeR, 'typeF1' : typeF1,
#                        'subtypeP' : subtypeP, 'subtypeR' : subtypeR, 'subtypeF1' : subtypeF1})

def writeout(SID, corpus, predictions, probs, revs, idMapping, idx2word, idx2label, ofile):

    counter = -1
    holder = defaultdict(list)
    for id, pred, prob in zip(corpus['id'], predictions, probs):
    
        if pred == 0: continue
    
        counter += 1
        if id not in idMapping:
            print 'cannot find id : ', id , ' in mapping'
            exit()
        ikey = idMapping[id]
        keyls = ikey.split()
        doc = keyls[1]
        instanceId = int(keyls[2])
        
        start = revs[doc]['instances'][instanceId]['wordStart']
        end = revs[doc]['instances'][instanceId]['wordEnd']
        anchor = revs[doc]['instances'][instanceId]['anchor']
        word = revs[doc]['instances'][instanceId]['word'][anchor]
        if pred not in idx2label:
            print 'cannot find prediction: ', pred, ' in idx2label'
            exit()
        subtype = idx2label[pred]
        subTypeConfidence = numpy.max(prob)
        
        if subtype not in subtype2fullType:
            print 'cannot find subtype in subtype2fullType: ', subtype
            exit()
        
        out = SID
        out += '\t' + doc
        out += '\t' + 'E' + str(counter)
        out += '\t' + str(start) + ',' + str(end)
        out += '\t' + word
        out += '\t' + subtype2fullType[subtype]
        out += '\t' + 'NONE'
        out += '\t' + '1.0'
        out += '\t' + str(subTypeConfidence)
        out += '\t' + '1.0'
        
        outId = out + '\t' + str(instanceId)
        
        holder[doc] += [(out, outId)]
        
        out, outId = considerBothDieAndAttack(outId, counter)
        if out and outId:
            holder[doc] += [(out, outId)]
            counter += 1            
    
    writer = open(ofile, 'w')
    writerId = open(ofile + '.id', 'w')
    for doc in holder:
        writer.write('#BeginOfDocument ' + doc + '\n')
        writerId.write('#BeginOfDocument ' + doc + '\n')
        for em, emid in holder[doc]:
            writer.write(em + '\n')
            writerId.write(emid + '\n')
        for i, em in enumerate(holder[doc]):
            id = em[1].split('\t')[2]
            writer.write('@Coreference' + '\t' + 'C' + str(i) + '\t' + id + '\n')
            writerId.write('@Coreference' + '\t' + 'C' + str(i) + '\t' + id + '\n')
        writer.write('#EndOfDocument' + '\n')
        writerId.write('#EndOfDocument' + '\n')
    writer.close()
    writerId.close()
    
def considerBothDieAndAttack(line, counter):
    els = line.split()
    if els[4].lower() not in bothDieAndAttack: return None, None
    newType = None
    if els[5] == 'Conflict_Attack': newType = 'Life_Die'
    if els[5] == 'Life_Die': newType = 'Conflict_Attack'
    if not newType: return None, None
    els[5] = newType
    els[2] = 'E' + str(counter+1)
    
    return '\t'.join(els[0:-1]), '\t'.join(els)

def myScore(goldFile, systemFile, tokenFile):

    proc = subprocess.Popen(["python", scoreScript, "-g", goldFile, "-s", systemFile, "-t", tokenFile], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    ous, _ = proc.communicate()
    
    spanP, spanR, spanF1 = 0.0, 0.0, 0.0
    subtypeP, subtypeR, subtypeF1 = 0.0, 0.0, 0.0
    startStoring = False
    for line in ous.split('\n'):
        line = line.strip()
        if line == '=======Final Mention Detection Results=========':
            startStoring = True
            continue
        if not startStoring: continue
        if line.startswith('plain'):
            els = line.split('\t')
            spanP, spanR, spanF1 = float(els[1]), float(els[2]), float(els[3])
            continue
        if line.startswith('mention_type') and not line.startswith('mention_type+realis_status'):
            els = line.split('\t')
            subtypeP, subtypeR, subtypeF1 = float(els[1]), float(els[2]), float(els[3])
            continue 
    
    return OrderedDict({'spanP' : spanP, 'spanR' : spanR, 'spanF1' : spanF1,
                        'typeP' : subtypeP, 'typeR' : subtypeR, 'typeF1' : subtypeF1,
                        'subtypeP' : subtypeP, 'subtypeR' : subtypeR, 'subtypeF1' : subtypeF1})

def getPerformance(total, predicted, correct):
    p = 0.0 if predicted == 0 else 1.0 * correct / predicted
    r = 0.0 if total == 0 else 1.0 * correct / total
    f1 = 0.0 if (p + r) == 0 else (2*p*r) / (p+r)
    return p, r, f1

def readAnnotationFile(afile):
    typeRes = defaultdict(lambda : defaultdict(set))
    subtypeRes = defaultdict(lambda : defaultdict(set))
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
            
            typeRes[doc][span].add(etype) # += [etype]
            subtypeRes[doc][span].add(st) # += [st]
    return typeRes, subtypeRes

def saving(corpus, predictions, probs, idx2word, idx2label, idMapping, address):
        
    def generateEvtSent(rid, sent, anchor, start, end, pred, golden, idx2word, idx2label, idMapping):
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
        
        prediction = idx2label[pred].lower()
        golden = idx2label[golden].lower()
              
        res += '\t' + 'pred=' + prediction + '\t' + 'golden=' + golden
        res += '\t' + ('__TRUE__' if prediction == golden else '__FALSE__')
        
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
    

    for rid, sent, anchor, start, end, pred, golden, pro in zip(corpus['id'], corpus['word'], corpus['position'], corpus['wordStart'], corpus['wordEnd'], predictions, corpus['label'], probs):
        fout.write(generateEvtSent(rid, sent, anchor, start, end, pred, golden, idx2word, idx2label, idMapping) + '\n')
        fprobOut.write(generateProb(rid, pro, start, end, idx2label, idMapping) + '\n')
    
    fout.close()
    fprobOut.close()
    
def produceBinaryTokenFeatures(oldBatch, newBatch, index):
    newBatch[::] = 0.0
    for o, n in zip(oldBatch, newBatch):
        n[index, o+1] = 1.0
    return newBatch[:,:,1:]

def generateParameterFileName(model, expected_features, nhidden, conv_feature_map, conv_win_feature_map, multilayerNN1, multilayerNN2):
    res = model + '.f-'
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

def train(SID='',
          dataset_path='',
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
    
    folder = '/scratch/thn235/projects/nugget/res/' + folder

    paramFolder = folder + '/params'

    if not os.path.exists(folder): os.mkdir(folder)
    if not os.path.exists(paramFolder): os.mkdir(paramFolder)
    
    paramFileName = paramFolder + '/' + generateParameterFileName(model, expected_features, nhidden, conv_feature_map, conv_win_feature_map, multilayerNN1, multilayerNN2)

    print 'loading dataset: ', dataset_path, ' ...'
    revs, embeddings, dictionaries = cPickle.load(open(dataset_path, 'rb'))
    
    idx2label = dict((k,v) for v,k in dictionaries['subtype'].iteritems())
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
    
    datasets, idMappings, binaryDimMap, binaryLenMap = make_data(revs, dictionaries, embeddings, features, window)
    
    dimCorpus = datasets['train']
    
    vocsize = len(idx2word)
    nclasses = len(idx2label)
    nsentences = len(dimCorpus['word'])

    print 'vocabsize = ', vocsize, ', nclasses = ', nclasses, ', nsentences = ', nsentences, ', word embeddings dim = ', emb_dimension
    
    features_dim = OrderedDict([('word', emb_dimension)])
    for ffin in expected_features:
        if features[ffin] == 1:
            features_dim[ffin] = binaryDimMap[ffin]
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
            datasets[corpus][ed] = numpy.array(datasets[corpus][ed], dtype='int32')
    
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
    
    binTokenFets, binTokenIndexes = {}, {}
    for fet in features:
        if features[fet] != 1: continue
        binTokenFets[fet] = numpy.zeros((batch, conv_winre, 1+binaryDimMap[fet]), dtype='float32')
        binTokenIndexes[fet] = numpy.zeros((conv_winre, binaryLenMap[fet]), dtype='int32')
        for ii in range(conv_winre): binTokenIndexes[fet][ii] = ii
    
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
            for ed in features:
                if features[ed] >= 0:
                    if ed not in trainCorpus:
                        print 'cannot find data in train for: ', ed
                        exit()
                    
                    trainIn[ed] = trainCorpus[ed][minibatch_index*batch:(minibatch_index+1)*batch]
                    if features[ed] == 1:
                        trainIn[ed] = produceBinaryTokenFeatures(trainIn[ed], binTokenFets[ed], binTokenIndexes[ed])

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
            _predictions[elu], _probs[elu] = predict(evaluatingDataset[elu], batch, reModel, idx2word, idx2label, features, binTokenFets, binTokenIndexes)
            
            writeout(SID, evaluatingDataset[elu], _predictions[elu], _probs[elu], revs[elu], idMappings[elu], idx2word, idx2label, folder + '/' + elu + '.pred' + str(e))
            
            if goldFiles[elu] and tokenFiles[elu]:
                _perfs[elu] = myScore(goldFiles[elu], folder + '/' + elu + '.pred' + str(e), tokenFiles[elu])
        
        perPrint(_perfs)
        
        print 'saving parameters ...'
        reModel.save(paramFileName + str(e) + '.pkl')
        
        print 'saving output ...'
        for elu in evaluatingDataset:
            saving(evaluatingDataset[elu], _predictions[elu], _probs[elu], idx2word, idx2label, idMappings[elu], folder + '/' + elu + str(e) + '.out')
        
        if _perfs['valid']['subtypeF1'] > best_f1:
            #rnn.save(folder)
            best_f1 = _perfs['valid']['subtypeF1']
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
    
    print '------------------------------------------------------------------------------'

if __name__ == '__main__':
    pass
