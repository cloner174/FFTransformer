#In the name of God##
import torch
import random
import numpy as np
from run import go_model
from utils.tools import dotdict


fix_seed = 2022
random.seed(fix_seed)
torch.manual_seed(fix_seed)

np.random.seed(fix_seed)

args = dotdict()

# basic config
args.is_training = 1                  #  hint: statu
args.model_id = 'test'                #  hint: model id for saving
args.model = 'IndRNN'          #  hint: model name, options: FFTransformer, Autoformer, LSTM, MLP, Informer,
#                                                          options: Transformer, LogSparse, persistence
#                                                          And same with GraphXxxx, like: GraphTransformer, GraphLSTM, and ..
args.plot_flag  =  1                  #  hint:  Whether to save loss plots or not
args.test_dir = 'custom/'                    #  hint:  Base dir to save test results
args.verbose = 1                      #  hint:  Whether to print inter-epoch losses

# data loader
args.data = 'Custom'                                              # hint: dataset type, Wind or WindGraph
args.root_path = 'custom/dataset/'          # hint:  root path of the data file
args.data_path = 'data.csv'                                # hint:  data file
args.target = 'Close'                                # hint:  optional target station for non-graph models
args.scale = True
args.freq = 'b'                                                 # hint:  freq for time features encoding:
#                                                                      options: [ s:secondly, t:minutely, h:hourly, 
#                                                                      options:   d:daily, b:business days, w:weekly, m:monthly]
#                                                                      options:   You can also use more detailed freq like 15min or 3h
args.checkpoints = 'custom/checkpoints/'                              # hint: location of model checkpoints
args.checkpoint_flag = 1                                        # hint: Whether to checkpoint or not
args.n_closest = None                                           # hint: number of closest nodes for graph connectivity, None --> complete graph
args.all_stations = 0                                           # hint: Whether to use all stations or just target for non-spatial models
args.data_step = 1                                              # hint: Only use every nth point. Set data_step  =  1 for full dataset
args.min_num_nodes = 2                                          # hint: Minimum number of nodes in a graph

# forecasting task
args.features = 'S'          # hint:  forecasting task, options:[M, S]; M:multivariate input, S:univariate input
args.seq_len = 5             # hint: input sequence length
args.label_len = 1           # hint: start token length. Note that Graph models only use label_len and pred_len
args.pred_len = 1            # hint: prediction sequence length
args.enc_in = 1              # hint: Number of encoder input features
args.dec_in = 1              # hint: Number of decoder input features
args.c_out = 1               # hint: output size, note that it is assumed that the target features are placed last

# model define
args.d_model = 512                           # hint: dimension of model
args.n_heads = 8                             # hint: num of heads
args.e_layers = 2                            # hint: number of encoder layers for non-spatial and number of LSTM or MLP layers for GraphLSTM and GraphMLP
args.d_layers = 1                            # hint: num of decoder layers
args.gnn_layers = 2                          # hint: Number of sequential graph blocks in GNN
args.d_ff = 2048                             # hint: dimension of fcn
args.moving_avg = 25                         # hint: window size of moving average for Autoformer
args.factor = 3                              # hint: attn factor
args.distil = True                           # hint: whether to use distilling in encoder, using this argument means not using distilling, not used for GNN models
args.dropout = 0.05                          # hint: dropout
args.embed = 'timeF'                         # hint = time features encoding, options:[timeF, fixed, learned]
args.activation = 'gelu'                     # hint:  activation
args.output_attention  =  False              # hint = whether to output attention in ecoder, action:store_true
args.win_len = 6                             # hint: Local attention length for LogSparse Transformer
args.res_len = None                          # hint: Restart attention length for LogSparse Transformer
args.qk_ker = 4                              # hint: Key/Query convolution kernel length for LogSparse Transformer
args.v_conv = 0                              # hint: Weather to apply ConvAttn for values (in addition to K/Q for LogSparseAttn)
args.sparse_flag = 1                         # hint: Weather to apply logsparse mask for LogSparse Transformer
args.top_keys = 0                            # hint: Weather to find top keys instead of queries in Informer
args.kernel_size = 3                         # hint: Kernel size for the 1DConv value embedding
args.train_strat_lstm = 'recursive'          # hint: The training strategy to use for the LSTM model. recursive or mixed_teacher_forcing
args.norm_out = 1                            # hint: Whether to apply laynorm to outputs of Enc or Dec in FFTransformer
args.num_decomp = 4                          # hint: Number of wavelet decompositions for FFTransformer
args.mlp_out = 0                             # hint: Whether to apply MLP to GNN outputs

# Optimization
args.num_workers = 1                  # hint: data loader num workers
args.itr = 1                          # hint: experiments times
args.train_epochs = 10                # hint: train epochs
args.batch_size = 32                  # hint: batch size of train input data
args.patience = 5                     # hint: early stopping patience
args.learning_rate  =  0.001          # hint: optimizer learning rate
args.lr_decay_rate  =  0.8            # hint: Rate for which to decay lr with
args.des = 'test'                     # hint:  exp description
args.loss = 'mse'                     # hint:  loss function
args.lradj = 'type1'                  # hint:  adjust learning rate

# GPU
args.use_gpu  =  True          # hint: use gpu
args.gpu  =  0                 # hint: gpu

# multi-gpu is not fully developed yet and still experimental for graph data.
args.use_multi_gpu = False          # hint: should use multiple gpus ? , action: store_true, 
args.devices = '0,1,2,3'            # hint: device ids of multiple gpus


# added options
args.criteria = 'default'               # hint: kind of measure for selecting criterion, options: 'SmoothL1', 'Huber', 'L1'
#                                            default is -> MSE   !  Cation! THe VaLUe FoR thIS PArT is CASE-SENSITIVE !
args.kind_of_optim = 'default'          # hint: kind of optimizer to use, default is Adam
args.kind_of_scaler = 'MinMax'



model, setting = go_model(args)
