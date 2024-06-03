import torch
import numpy as np
from exp.exp_main import Exp_Main



def go_model(args):
    
    Exp = Exp_Main
    
    if args.features == 'S':
        assert (np.array([args.c_out, args.enc_in, args.dec_in]) == 1).all(), "c_out, enc_in and dec_in should be 1 for univariate"
    
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]
    
    print('Args in experiment:')
    print(args)
    if args.is_training == 1:
        
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                ii)
            
            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, base_dir=args.test_dir)
            
            torch.cuda.empty_cache()
    else:
        
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            ii)
        
        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()