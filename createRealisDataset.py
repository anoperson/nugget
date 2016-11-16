import numpy
import sys
import subprocess
import os
import random
import cPickle
import copy

from collections import OrderedDict, defaultdict

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

def make_data(revs, dictionaries, embeddings, outPath):

    idx2label = dict((k,v) for v,k in dictionaries['subtype'].iteritems())
    idx2realis = dict((k,v) for v,k in dictionaries['realis'].iteritems())
    
    for datn in revs:
        writer = open(outPath + '/' + datn + '.realis', 'w')
        for doc in revs[datn]:
            writer.write('#BeginOfDocument ' + doc + '\n')
            instanceId = -1
            for rev in revs[datn][doc]['instances']:
                
                instanceId += 1
                subtype = int(rev['subtype'])
                if subtype == 0: continue
                
                if subtype not in idx2label:
                    print 'cannot find subtype: ', subtype, ' in idx2label'
                    exit()
                subtype = idx2label[subtype]
                if subtype not in subtype2typeMap:
                    print 'cannot find subtype: ', subtype, ' in mapping'
                    exit()
                type = subtype2typeMap[subtype]
                
                start = rev['wordStart']
                end = rev['wordEnd']
                anchor = rev['anchor']
                word = rev['word'][anchor]
                
                realis = rev['realis']
                if realis not in idx2realis:
                    print 'cannot find realis id : ', realis, ' in idx2realis'
                    exit()
                realis = idx2realis[realis]
                
                eventId = rev['eventId']
                eventId = eventId[3:]
        
                out = 'NYU'
                out += '\t' + doc
                out += '\t' + 'E' + str(eventId)
                out += '\t' + str(start) + ',' + str(end)
                out += '\t' + word
                out += '\t' + type + '_' + subtype
                out += '\t' + realis
                out += '\t' + '1.0'
                out += '\t' + '1.0'
                out += '\t' + '1.0'
        
                out += '\t' + str(instanceId)
                
                writer.write(out + '\n')
            for i, chain in enumerate(revs[datn][doc]['coreference']):
                writer.write('@Coreference' + '\t' + 'C' + str(i) + '\t' + convertChain(chain) + '\n')
                
            writer.write('#EndOfDocument' + '\n')
            
        writer.close()

def convertChain(chain):
    res = ''
    rchain = [ 'E' + em[3:] for em in chain ]
    return ','.join(rchain)

def main():
    dataset_path = sys.argv[1]
    outPath = sys.argv[2]
    revs, embeddings, dictionaries = cPickle.load(open(dataset_path, 'rb'))
    
    make_data(revs, dictionaries, embeddings, outPath)

if __name__ == '__main__':
    main()
