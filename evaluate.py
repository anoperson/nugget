from nugget import train
from collections import OrderedDict
import sys

def main(params):
    print params
    train(model = params['model'],
          wedWindow = params['wedWindow'],
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

if __name__=='__main__':
    pars={'model' : 'convolute', # convolute # rnnHead, rnnMax, rnnHeadFf, rnnMaxFf, rnnHeadForward, rnnHeadBackward, rnnMaxForward, rnnMaxBackward, rnnHeadFfForward, rnnHeadFfBackward, rnnMaxFfForward, rnnMaxFfBackward # alternateHead, alternateMax, alternateConv
          'wedWindow' : 2,
          'expected_features' : OrderedDict([('partOfSpeech', -1),
                                             ('chunk', -1),
                                             ('entity', 0),
                                             ('dependency', -1),
                                             ('dist1', 0),
                                             ('dist2', 0)]),
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
    prefix = sys.argv[4]
    folder = 'mo_' + pars['tmode'] \
             + '.wi_' + str(pars['wedWindow']) \
             + '.model_' + pars['model'] \
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
    pars['folder'] =  folder
    main(pars)
