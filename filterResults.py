import sys
from collections import defaultdict

acceptType = ['Conflict_Attack',
              'Conflict_Demonstrate',
              'Contact_Broadcast',
              'Contact_Contact',
              'Contact_Correspondence',
              'Contact_Meet',
              'Justice_Arrest-Jail',
              'Life_Die',
              'Life_Injure',
              'Manufacture_Artifact',
              'Movement_Transport-Artifact',
              'Movement_Transport-Person',
              'Personnel_Elect',
              'Personnel_End-Position',
              'Personnel_Start-Position',
              'Transaction_Transaction',
              'Transaction_Transfer-Money',
              'Transaction_Transfer-Ownership']

subtype2fullType = {"Business_Declare-Bankruptcy" : "business.declareBankruptcy",

                    "Manufacture_Artifact" : "manufacture.artifact",
                   
                    "Personnel_Start-Position" : "personnel.startPosition",
                    "Personnel_End-Position" : "personnel.endPosition",
                    "Personnel_Nominate" : "personnel.nominate",
                    "Personnel_Elect" : "personnel.elect",
                   
                    "Conflict_Demonstrate" : "conflict.demonstrate",
                    "Conflict_Attack" : "conflict.attack",
                   
                    "Contact_Broadcast" : "contact.broadcast",
                    "Contact_Contact" : "contact.contact",
                    "Contact_Correspondence" : "contact.correspondence",
                    "Contact_Meet" : "contact.meet",
                   
                    "Transaction_Transfer-Money" : "transaction.transferMoney",
                    "Transaction_Transfer-Ownership" : "transaction.transferOwnership",
                    "Transaction_Transaction" : "transaction.transaction",
                   
                    "Movement_Transport-Artifact" : "movement.transportartifact",
                    "Movement_Transport-Person" : "movement.transportperson",
                   
                    "Business_Start-Org" : "business.startOrg",
                    "Business_End-Org" : "business.endOrg",
                    "Business_Merge-Org" : "business.mergeOrg",
                   
                    "Life_Die" : "life.die",
                    "Life_Divorce" : "life.divorce",
                    "Life_Marry" : "life.marry",
                    "Life_Beborn" : "life.beBorn",
                    "Life_Injure" : "life.injure",
                   
                    "Justice_Pardon" : "justice.pardon",
                    "Justice_Sue" : "justice.sue",
                    "Justice_Convict" : "justice.convict",
                    "Justice_Charge-Indict" : "justice.chargeIndict",
                    "Justice_Trial-Hearing" : "justice.trialHearing",
                    "Justice_Sentence" : "justice.sentence",
                    "Justice_Appeal" : "justice.appeal",
                    "Justice_Release-Parole" : "justice.releaseParole",
                    "Justice_Extradite" : "justice.extradite",
                    "Justice_Fine" : "justice.fine",
                    "Justice_Execute" : "justice.execute",
                    "Justice_Arrest-Jail" : "justice.arrestJail",
                    "Justice_Acquit" : "justice.acquit"}

def main():
    sid = sys.argv[1]
    ifile = sys.argv[2]
    ofile = sys.argv[3]
    eid = -1
    cid = -1
    
    idMap = {}
    removedIds = set()
    
    writer = open(ofile, 'w')
    
    currentDoc = ''
    
    stats = defaultdict(int)
    
    with open(ifile, 'r') as f:
        for line in f:
            line = line.strip()
            
            if line.startswith('#BeginOfDocument'):
                currentDoc = line.split()[1]
                cid = -1
                idMap = {}
                removedIds = set()
                writer.write(line + '\n')
                continue
            
            if line == '#EndOfDocument':
                print 'removed: ', len(removedIds), ' events from', currentDoc
                stats['removed'] += len(removedIds)
                writer.write(line + '\n')
                continue
            
            els = line.split('\t')
            
            if line.startswith('@Coreference'):
                chain = els[2].split(',')
                kept = True
                for c in chain:
                    if c in removedIds:
                        kept = False
                        break
                if kept:
                    newChain = []
                    for c in chain:
                        if c not in idMap:
                            print 'cannot find id in map: ', c
                            exit()
                        newChain.append(idMap[c])
                    newChain = ','.join(newChain)
                    cid += 1
                    rid = 'C' + str(cid)
                    els[1] = rid
                    els[2] = newChain
                    writer.write('\t'.join(els) + '\n')
                continue
            
            els[0] = sid
            els[7] = els[8]
            
            ieid = els[2]
            itype = els[5]
            
            if itype not in acceptType:
                removedIds.add(ieid)
                continue
            
            eid += 1
            els[2] = 'E' + str(eid)
            idMap[ieid] = els[2]
            
            stats[itype] += 1
            stats['total'] += 1
            
            if els[5] not in subtype2fullType:
                print 'cannot find type in map: ', els[5]
                exit()
            
            els[5] = subtype2fullType[els[5]]
            els[6] = els[6].upper()
            
            writer.write('\t'.join(els) + '\n')
    
    writer.close()
    
    coid = 0
    print '-------------------Detected--------------'
    for t in stats:
        if t == 'total' or t == 'removed': continue
        coid += 1
        print coid, ' -> ', t, ' : ', stats[t]
    print '----------'
    print 'total : ', stats['total']
    
    print 'removed: ', stats['removed']

if __name__ == '__main__':
    main()