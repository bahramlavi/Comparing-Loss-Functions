import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.constraints import MaxNorm
from tensorflow.keras.optimizers.legacy import Adam, SGD
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
from scikeras.wrappers import KerasRegressor

# Initializing the models hyperparameters treating as a set of continues values

# number of neurons for use of lstm
neurons = [128]#, 50, 100]

# 1st dense layer
neurons1 = [128]
# 2nd dense layer
neurons2 = [5]

# Optimizers for grid-search
optimizer = [Adam()]#, SGD()]


# some extra parameters for DL methods
weight_constraint = [0.1, 0.3, 1.0]
dropout_rate = [0.25]
activation = 'tanh'#,'tanh']
batch_size = [32]
epochs = [50]
tf.reduce_retracing = True

device = '/GPU:0'
    
def call_DL_Regressor():
    def get_gridsearch_with_LSTM(train_X, test_X, train_y, X_val,y_val, loss):
        # Function to create model, required for KerasClassifier
        optimizer = Adam(learning_rate=0.001, clipnorm=3.0)
        def create_model(neurons):
            # create model
            model = Sequential()
            model.add(LSTM(neurons, input_shape=(train_X.shape[1], train_X.shape[2]), activation=activation))
            model.add(Dense(1,activation=activation))
            model.compile(optimizer=optimizer, loss=loss) 
            return model

        # time series inside gridsearch
        #tscv = TimeSeriesSplit(n_splits=num_split)
        model = KerasRegressor(build_fn=create_model,verbose=0, shuffle=False)

        param_grid = dict(model__neurons=neurons)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=1,
                            cv=None,
                            error_score='raise')  # accuracy

        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y, validation_data=(X_val, y_val))
        train_end = datetime.datetime.now()

        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)

        reg_f = grid_result.best_estimator_
        #oilPred = reg_f.predict(train_X)
        #oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start), reg_f


   
    

    def get_gridsearch_with_LSTMbidirectional(train_X, test_X, train_y, X_val,y_val,loss):
        # Function to create model, required for KerasClassifier
        optimizer = Adam(learning_rate=0.001, clipnorm=3.0)
        def create_model(neurons):
            # create model
            model = Sequential()
            model.add(
                Bidirectional(LSTM(neurons,activation=activation), input_shape=(train_X.shape[1], train_X.shape[2])))  # return_sequences=True
            model.add(Dense(1))
            model.compile(optimizer=optimizer, loss=loss, metrics=['mse']) 
            return model

        # time series inside gridsearch
        #tscv = TimeSeriesSplit(n_splits=num_split)
        model = KerasRegressor(build_fn=create_model, verbose=0, shuffle=False)

        param_grid = dict(model__neurons=neurons)

        grid = GridSearchCV(estimator=model,
                            param_grid=param_grid,
                            scoring='neg_mean_absolute_error',
                            n_jobs=1,
                            cv=None,
                            error_score='raise')  # accuracy

        train_start = datetime.datetime.now()
        with tf.device(device):
            grid_result = grid.fit(train_X, train_y, validation_data=(X_val, y_val))
        train_end = datetime.datetime.now()

        oilPred = grid_result.best_estimator_.predict(train_X)
        oilForecast = grid_result.best_estimator_.predict(test_X)
        
        reg_f = grid_result.best_estimator_
        #oilPred = reg_f.predict(train_X)
        #oilForecast = reg_f.predict(test_X)
        
        return oilPred,oilForecast, (train_end - train_start), reg_f


    regs = {'LSTM': get_gridsearch_with_LSTM,
            'BiLSTM': get_gridsearch_with_LSTMbidirectional
            }
    return regs



