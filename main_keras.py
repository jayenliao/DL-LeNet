'''
Deep Learning - HW4: LeNet-5 (Keras verion)
Jay Liao (re6094028@gs.ncku.edu.tw)
'''

import time, os
import numpy as np
import pandas as pd
from lenet_keras.args import init_arguments
from lenet_keras.utils import *
from lenet_keras.trainer import Trainer

def main(args, return_trainer=False):
    
    # ---- (1) Load and prepeocessing ---- #

    PATH = args.dataPATH if args.dataPATH[-1] == '/' else args.dataPATH + '/'
    
    X_tr, y_tr = get_resized_data('train', PATH, args.resize)
    X_va, y_va = get_resized_data('val', PATH, args.resize)
    X_te, y_te = get_resized_data('test', PATH, args.resize)

    print('\nShapes of feature matrices (Train | Val | Test):')
    print(X_tr.shape)
    print(X_va.shape)
    print(X_te.shape)
    print('\nShapes of y label matrices (Train | Val | Test):')
    print(y_tr.shape, y_va.shape, y_te.shape)
    
    # ---- (2) Training ---- #
    # ---- (3) Evaluating ---- #

    d_trainers = {}
    trainer = Trainer(
        X_tr, y_tr, model_name=args.model_name, 
        input_shape=X_tr.shape[1:],
        filter_size=args.filter_size,
        hidden_act=args.hidden_act,
        hidden_sizes=args.hidden_sizes,
        pooling_size=args.pooling_size,
        strides=args.strides, lr=args.lr,
        output_size=one_hot_transformation(y_tr).shape[1],
        savePATH=args.savePATH,
        verbose=args.verbose,
        random_state=args.random_state
    )  
    trainer.train(
        X_va=X_va, y_va=y_va,
        X_te=X_te, y_te=y_te,
        loss_fun=args.loss_fun,
        metrics=args.eval_metrics,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=args.verbose,
        workers=args.n_jobs
    )

    trainer.plot_training('loss', args.plot_figsize)
    trainer.plot_training('accuracy', args.plot_figsize)
    print('')

    if return_trainer:
        d_trainers['trainer'] = trainer
        return d_trainers

if __name__ == '__main__':
    args = init_arguments().parse_args()
    main(args)