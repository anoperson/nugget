from nuggetRealis import train
from collections import OrderedDict
import sys
import argparse

def main(params):
    print params
    train(dataset_path = params['dataset_path'],
          eventTypePath = params['eventTypePath'],
          model = params['model'],
          wedWindow = params['wedWindow'],
          window = params['window'],
          expected_features = params['expected_features'],
          givenPath = params['givenPath'],
          withEmbs = params['withEmbs'],
          updateEmbs = params['updateEmbs'],
          optimizer = params['optimizer'],
          lr = params['lr'],
          dropout = params['dropout'],
          regularizer = params['regularizer'],
          norm_lim = params['norm_lim'],
          verbose = params['verbose'],
          decay = params['decay'],
          batch = params['batch'],
          binaryCutoff = params['binaryCutoff'],
          multilayerNN1 = params['multilayerNN1'],
          multilayerNN2 = params['multilayerNN2'],
          nhidden = params['nhidden'],
          conv_feature_map = params['conv_feature_map'],
          conv_win_feature_map = params['conv_win_feature_map'],
          seed = params['seed'],
          #emb_dimension=300, # dimension of word embedding
          nepochs = params['nepochs'],
          folder = params['folder'])
def fetStr(ef):
    res = ''
    for f in ef:
        res += str(ef[f])
    return res

def fmStr(ft):
    res = ''
    for f in ft:
        res += str(f)
    return res

def argsp():
    aparser = argparse.ArgumentParser()
    
    aparser.add_argument('--dataset_path', help='path to the pkl dataset file')
    aparser.add_argument('--eventTypePath', help='path to detected event instances with types')
    aparser.add_argument('--model', help='model to be used, see the code for potential options')
    aparser.add_argument('--wedWindow', help='window for local context (concatenation of surrouding emeddings)', type=int)
    aparser.add_argument('--window', help='window for instances', type=int)
    aparser.add_argument('--givenPath', help='path to the trained model parameters to initialize')
    aparser.add_argument('--withEmbs', help='using word embeddings to intialize or not')
    aparser.add_argument('--updateEmbs', help='update embeddings during training or not')
    aparser.add_argument('--optimizer', help='optimier to use for training')
    aparser.add_argument('--lr', help='learning rate', type=float)
    aparser.add_argument('--dropout', help='dropout rate', type=float)
    aparser.add_argument('--regularizer', help='regularizer rate', type=float)
    aparser.add_argument('--norm_lim', help='normalization constant', type=float)
    aparser.add_argument('--verbose', help='print more info or not', type=int)
    aparser.add_argument('--decay', help='decay or not')
    aparser.add_argument('--batch', help='number of instances per batch', type=int)
    aparser.add_argument('--binaryCutoff', help='cutoff for binary features', type=int)
    aparser.add_argument('--multilayerNN1', help='dimensions for the fist multiplayer neural nets', type=int, nargs='*')
    aparser.add_argument('--multilayerNN2', help='dimensions for the second multiplayer neural nets', type=int, nargs='*')
    aparser.add_argument('--nhidden', help='number of hidden units', type=int)
    aparser.add_argument('--conv_feature_map', help='number of filters for convolution', type=int)
    aparser.add_argument('--conv_win_feature_map', help='windows for filters for convolution', type=int, nargs='+')
    aparser.add_argument('--seed', help='random seed', type=int)
    aparser.add_argument('--nepochs', help='number of iterations to run', type=int)
    
    aparser.add_argument('--anchor', help='features : anchor', type=int)
    aparser.add_argument('--pos', help='features : pos', type=int)
    aparser.add_argument('--chunk', help='features : chunk', type=int)
    aparser.add_argument('--possibleTypes', help='features : possibleTypes', type=int)
    aparser.add_argument('--dep', help='features : dep', type=int)
    aparser.add_argument('--nonref', help='features : nonref', type=int)
    aparser.add_argument('--title', help='features : title', type=int)
    aparser.add_argument('--eligible', help='features : eligible', type=int)
    
    
    return aparser

if __name__=='__main__':
    
    pars={'dataset_path' : '/misc/proteus108/thien/projects/fifth/eventNugget/nn/eligible0.win31.word2vec_nugget.pkl',
          'eventTypePath' : '',
          'model' : 'convolute', # convolute # rnnHead, rnnMax, rnnHeadFf, rnnMaxFf, rnnHeadForward, rnnHeadBackward, rnnMaxForward, rnnMaxBackward, rnnHeadFfForward, rnnHeadFfBackward, rnnMaxFfForward, rnnMaxFfBackward # alternateHead, alternateMax, alternateConv, nonConsecutiveConvolute
          'wedWindow' : 2,
          'window' : 31,
          'expected_features' : OrderedDict([('anchor', 0),
                                             ('pos', -1),
                                             ('chunk', -1),
                                             ('possibleTypes', -1),
                                             ('dep', 1),
                                             ('nonref', -1),
                                             ('title', -1),
                                             ('eligible', -1),]),
          'givenPath' : None,                          
          'withEmbs' : True,
          'updateEmbs' : True,
          'optimizer' : 'adadelta',
          'lr' : 0.01,
          'dropout' : 0.5,
          'regularizer' : 0.0,
          'norm_lim' : 9.0,
          'verbose' : 1,
          'decay' : False,
          'batch' : 50,
          'binaryCutoff' : -1,
          'multilayerNN1' : [],
          'multilayerNN2' : [],
          'nhidden' : 300,
          'conv_feature_map' : 150,
          'conv_win_feature_map' : [2,3,4,5],
          'seed' : 3435,
          'nepochs' : 20,
          'folder' : './res'}
    
    args = vars(argsp().parse_args())
    for arg in args:
        if args[arg] != None:
            if arg == 'withEmbs' or arg == 'updateEmbs' or arg == 'decay':
                args[arg] = False if args[arg] == 'False' else True
            if arg in pars['expected_features']:
                if args[arg] != 0 and args[arg] != 1: args[arg] = -1
                print '*****Updating feature parameters: ', arg, '(', pars['expected_features'][arg], ' --> ', args[arg], ')'
                pars['expected_features'][arg] = args[arg]
                continue
            print '*****Updating default parameter: ', arg, '(', pars[arg], ' --> ', args[arg], ')'
            pars[arg] = args[arg]      
    
    folder = 'model_' + pars['model'] \
             + '.wi_' + str(pars['wedWindow']) \
             + '.iwi_' + str(pars['window']) \
             + '.h_' + str(pars['nhidden']) \
             + '.embs_' + str(pars['withEmbs']) \
             + '.upd_' + str(pars['updateEmbs']) \
             + '.batch_' + str(pars['batch']) \
             + '.cut_' + str(pars['binaryCutoff']) \
             + '.mul1_' + fmStr(pars['multilayerNN1']) \
             + '.mul2_' + fmStr(pars['multilayerNN2']) \
             + '.opt_' + pars['optimizer'] \
             + '.drop_' + str(pars['dropout']) \
             + '.reg_' + str(pars['regularizer']) \
             + '.fet_' + fetStr(pars['expected_features']) \
             + '.cvft_' + str(pars['conv_feature_map']) \
             + '.cvfm_' + fmStr(pars['conv_win_feature_map']) \
             + '.lr_' + str(pars['lr']) \
             + '.norm_' + str(pars['norm_lim']) \
             + '.s_' + str(pars['seed'])
    if pars['givenPath']: folder += '.gp'
    pars['folder'] =  'realis.' + folder
    
    main(pars)
