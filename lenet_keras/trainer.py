import sys, os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime
from tqdm import tqdm
import keras
from keras import optimizers, layers

class Trainer:
    def __init__(self, X_tr, y_tr, model_name:str, input_shape:tuple, filter_size:int, hidden_act:str, hidden_sizes:list, pooling_size:int, strides:tuple, output_size:int, lr:float, savePATH:str, verbose:bool, random_state=4028):
        hidden_act = hidden_act.lower()
        self.model_name = model_name
        self.model = keras.Sequential()
        self.model.add(layers.Conv2D(filters=6, kernel_size=filter_size, activation=hidden_act, input_shape=input_shape))
        self.model.add(layers.MaxPool2D(pool_size=pooling_size, strides=strides))
        self.model.add(layers.Conv2D(filters=16, kernel_size=filter_size, activation=hidden_act))
        self.model.add(layers.MaxPool2D(pool_size=pooling_size, strides=strides))
        if model_name.lower() == 'improved_lenet5':
            self.model.add(layers.Conv2D(filters=16, kernel_size=filter_size, activation=hidden_act))
            self.model.add(layers.MaxPool2D(pool_size=pooling_size, strides=strides))
        self.model.add(layers.Flatten())
        self.model.add(layers.Dense(units=hidden_sizes[0], activation=hidden_act))
        self.model.add(layers.Dense(units=hidden_sizes[1], activation=hidden_act))
        self.model.add(layers.Dense(units=output_size, activation='softmax'))

        self.X_tr = X_tr
        self.y_tr = y_tr
        #self.num_conv_layers = model_hyper_param['num_conv_layers']
        self.hidden_act = hidden_act
        self.hidden_sizes = hidden_sizes
        self.filter_size = filter_size
        self.pooling_size = pooling_size
        self.output_size = output_size

        self.train_size = self.Ｘ_tr.shape[0]
        self.savePATH = savePATH if savePATH[-1] == '/' else savePATH + '/'
        #self.pretrained_model = pretrained_model
        self.random_state = random_state
        self.verbose = verbose
        self.optimizer = optimizers.Adam(lr=lr)

    def train(self, X_va, y_va, X_te, y_te, loss_fun:str, metrics:str, batch_size:int, epochs:int, verbose:int, workers:int):
        t0 = time.time()
        self.epochs = epochs
        self.model.compile(optimizer=self.optimizer, loss=loss_fun, metrics=[metrics, keras.metrics.TopKCategoricalAccuracy()])
        self.history = self.model.fit(
            x=self.X_tr, y=self.y_tr, validation_data=(X_va, y_va),
            batch_size=batch_size, shuffle=True, epochs=epochs,
            verbose=verbose, workers=workers
        )

        # Report that the model have been finished training and 
        # Save the model performances: acc_tr/acc_va, acc_te, best_acc, and time_cost
        tdiff = time.time() - t0
        print('\nFinish training! Total time cost for %3d epochs: %.2f s' % (self.epochs, tdiff))
        print('\nEvaluate the trained model on the testing set ...')
        self.evaluation(X_te, y_te)

        self.dt = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
        self.folder_name = self.savePATH + self.dt + '_' + self.model_name + '_' + self.hidden_act + '_fs=' + str(self.filter_size) + '_bs=' + str(batch_size) + '_epochs=' + str(epochs) + '/'
        try:
            os.makedirs(self.folder_name)
        except FileExistsError:
            pass
        print('The model is save as:')
        fn = self.folder_name + 'lenet5_model.h5'
        self.model.save(fn)
        print(fn)

        print('\nModel performances are saved as the following files:')

        self.fn = self.folder_name + 'Accuracy.txt'
        fn = self.fn.replace('Acc', 'TestAcc')
        pd.DataFrame(self.accuracy_te, index=['score']).T.to_csv(fn)
        print('-->', fn)
        
        '''
        fn = self.fn.replace('Acc', 'BestAcc')
        pd.DataFrame(self.best_acc, index=[0]).T.to_csv(fn)
        if not self.verbose:
            print('-->', fn)
        
        fn = self.fn.replace('Accuracy', 'TimeCost')
        np.savetxt(fn, np.array(time_cost))
        if not self.verbose:
            print('-->', fn)
        '''

    def evaluation(self, X_te, y_te):
        y_pred = self.model.predict(X_te)
        self.accuracy_te = {
            'Top-1': self.accuracy_score(y_pred, y_te, 1),
            'Top-5': self.accuracy_score(y_pred, y_te, 5),
        }

    def accuracy_score(self, y, yt, top=1):
        try:
            y = y.cpu().numpy()
            yt = yt.cpu().numpy()
        except:
            pass

        if yt.ndim != 1:
            yt = np.argmax(yt, axis=1)
        if top == 1:
            y = np.argmax(y, axis=1)
            acc = np.array(y == yt).mean()
        else:
            y = np.argsort(y, axis=1)[:,-top:]
            lst = []
            for i in range(len(yt)):
                lst.append(yt[i] in y[i,:])
            acc = np.array(lst).mean()
        return acc

    def plot_training(self, type_:str, figsize:tuple, save_plot=True):
        plt.figure(figsize=figsize)
        x = np.arange(len(self.history.epoch))
        plt.plot(x, self.history.history[type_], label=type_+'(train)')
        plt.plot(x, self.history.history['val_'+type_], label=type_+'(val)')
        if type_ == 'loss':
            plt.ylabel('Loss')
        elif type_ == 'accuracy':
            plt.plot(x, self.history.history['top_k_categorical_accuracy'], label='Top-5 (train)')
            plt.plot(x, self.history.history['val_top_k_categorical_accuracy'], label='Top-5 (val)')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
        plt.legend()
        plt.title('Plot of' + type_.capitalize() + ' of ' + self.model_name)
        plt.xlabel('Epoch')
        plt.grid()

        if save_plot:
            fn = self.fn.replace('Accuracy', type_).replace('.txt', '.png')
            plt.savefig(fn)
            print('The', type_, 'plot is saved as', fn)