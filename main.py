from warnings import filterwarnings

filterwarnings('ignore')
import os

os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from pathlib import Path
from tensorflow.keras import backend as K

import tensorflow as tf
tf.keras.backend.set_floatx('float64')
seed=123
np.random.seed(seed)
tf.random.set_seed(seed)

# path to the raw well data
path_hist_data = './U4_Data/'

# path to the experimental outpus
path2main_output_folder = './Experiments/'
Path(path2main_output_folder).mkdir(parents=True, exist_ok=True)
path2signal = 'Signals/'
Path(path2main_output_folder+path2signal).mkdir(parents=True, exist_ok=True)
path2csvs = 'CSVs/'
Path(path2main_output_folder+path2csvs).mkdir(parents=True, exist_ok=True)

#To forecast oil rate for a given production well data (e.g., P11, P12)
attribute_names = ['ORp'] 

history_day_end = 2891
atts_name = ''
for i in range(len(attribute_names)):
    atts_name += attribute_names[i] + '_' if i + 1 < len(attribute_names) else attribute_names[i]

# time-lagged range
#lag_sizes = [10, 20, 50, 100]
lag_sizes = [2]#4,6]#, 20]#,100]
#portion of features selected with respect to lag size

periods_h = [7,30]#, 90]



#Portion of trainable data
n_trainable_data = 0.8

# experimental evaluation on the well producers P11 and P12
well_id = range(11, 13)
well_state = ['P']
                   

df_result_column_names = ['Regressor',
                          'Well',
                          'lag',
                          'period',
                          'loss',
                          'wmse_h',
                          'wmse_v',
                          'wmse_f',
                          'relDiff_h',
                          'relDiff_v',
                          'relDiff_f',
                          'NQDS_h',
                          'NQDS_v',
                          'NQDS_f',
                          'MAE_h',
                          'MAE_v',
                          'MAE_f',
                          'WMAPE_h',
                          'WMAPE_v',
                          'WMAPE_f',
                          'Proc. Time']

def compute_NQDS(y, y_hat):
    tolorance = 2
    cons = 20
    AQD = np.sum((tolorance*y+cons)**2)
    SD = np.sum((y_hat-y))
    if SD != 0:
        QDS = (SD/np.abs(SD)) * np.sum((y_hat-y))**2
    else:
        QDS = np.sum((y_hat-y))**2
    #print(QDS,AQD, SD)
    NQDS = QDS/AQD
    
    return np.round(NQDS,2)



def data_handling(data, atts, model_prediction, t_lags=0, dah=1):
    
    def add_lags(df):
        for lg in range(1, t_lags+1):
            for att in atts:
                #print(att)
                col = f'{att}_lag{lg}'  # .format(att, lg)  #Creates a column name
                df.insert(len(df.columns), col, df[att].shift(lg))
        return df


    data.reset_index(inplace=True)
    data_final = add_lags(data)
    data_final.insert(len(data_final.columns), 'pred', pd.Series(data[model_prediction].shift(-dah)))
    
    data_final.dropna(inplace=True, axis=0)

    return data_final.drop(['index'], axis=1)

def calculate_mse(y_true, y_pred):
  weighted_squared_errors = tf.square(y_pred - y_true)
  mse = tf.reduce_mean(weighted_squared_errors)
  return mse

def train_DL_models():
    from DL_Regressors import call_DL_Regressor
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, RandomizedSearchCV
    from scikeras.wrappers import KerasRegressor
    from Loss_functions import Call_loss_functions
    import tensorflow as tf
    tf.reduce_retracing = True


    
    df_res = pd.DataFrame(columns=df_result_column_names)

    for w_idx, w in enumerate(well_id):
        print(f'******** Processing on Well_ID = {w} \n')

        #Read the production data history
        df=pd.read_csv(f'{path_hist_data}Production_data.csv')
        attribute_names = list(df.columns)
        for lg in lag_sizes:
            for prd in periods_h:
                df_main = data_handling(df[attribute_names].copy(), attribute_names, model_prediction=f'OR{w}', t_lags=lg, dah=prd)
                print(df_main.columns)
                values = df_main.values
                values = values.astype('float64')
                features = values[:, :-1]
                labels = values[:, -1]
                split_index = int(len(features) * n_trainable_data)

                train_X = features[:split_index, :]
                train_V = features[split_index:split_index+int((len(features)-split_index)*.5), :]
                test_X = features[split_index+int((len(features)-split_index)*.5):, :]  
                trainY = labels[:split_index]
                val_y = labels[split_index:split_index+int((len(features)-split_index)*.5)]
                test_y = labels[split_index+int((len(features)-split_index)*.5):]
                
                
                scaler = MinMaxScaler(feature_range=(0, 1))  # (0,1)
                trainX = scaler.fit_transform(train_X)
                train_V = scaler.transform(train_V)                    
                testX = scaler.transform(test_X)
                
                scalerLA = MinMaxScaler(feature_range=(0, 1))
                trainY_main = scalerLA.fit_transform(trainY.reshape(-1, 1))  
                val_y = scalerLA.transform(val_y.reshape(-1, 1))  
                
                trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
                testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))
                train_V = train_V.reshape((train_V.shape[0], 1, train_V.shape[1]))

                
                regs = call_DL_Regressor()
                for reg in regs.keys():
                    loss_configurations = Call_loss_functions()
                
                    for i, config in enumerate(loss_configurations):
                        loss_short_name=config['short_name']
                        loss_name = config['name']
                        loss_fn = config['function']
                        
                        print(f'Well {w}, Regressor = {reg}, Lag = {lg}, Period = {prd}, loss={loss_short_name}')
                        if os.path.isfile(path2main_output_folder + path2signal + f'{w}_{reg}_{lg}_{prd}_{loss_short_name}.npy'):
                            df_res = pd.read_csv(path2main_output_folder + path2csvs + f'{atts_name}_dl_results.csv')
                            print(len(df_res))
                            continue  
                        
                        try:
                            oilPred_main, oilForecast, time_cost, reg_f = regs[reg](trainX, testX, trainY_main, train_V, val_y,loss_fn)
                            oilPred_validation = reg_f.predict(train_V)
                        
                            
                            oilPred_main=np.nan_to_num(oilPred_main.reshape(-1, 1), nan=0.0,  posinf=1e10, neginf=-1e10)
                            oilPred_validation=np.nan_to_num(oilPred_validation.reshape(-1, 1), nan=0.0,  posinf=1e10, neginf=-1e10)
                            oilForecast=np.nan_to_num(oilForecast.reshape(-1, 1), nan=0.0,  posinf=1e10, neginf=-1e10)    
                            
                            oilPred_main = scalerLA.inverse_transform(oilPred_main.reshape(-1, 1))
                            oilPred_validation = scalerLA.inverse_transform(oilPred_validation)
                            oilForecast = scalerLA.inverse_transform(oilForecast.reshape(-1,1))
                            
                            predData = np.vstack((oilPred_main, oilPred_validation, oilForecast))

                            trainY_main =  scalerLA.inverse_transform(trainY_main.reshape(-1, 1))
                            val_y =  scalerLA.inverse_transform(val_y)
                            
                            mse_h = calculate_mse(trainY_main, oilPred_main)
                            mse_v = calculate_mse(val_y, oilPred_validation)
                            mse_f = calculate_mse(test_y, oilForecast)
                            
                            mae_h = mean_absolute_error(trainY_main,oilPred_main)
                            mae_v = mean_absolute_error(val_y, oilPred_validation)
                            mae_f = mean_absolute_error(test_y, oilForecast)
                                    
                            
                            numerator = np.sum(np.abs(trainY_main - oilPred_main))
                            denominator = np.sum(np.abs(trainY_main))
                            wmape_h = numerator / denominator
                            
                            numerator = np.sum(np.abs(val_y - oilPred_validation))
                            denominator = np.sum(np.abs(val_y))
                            wmape_v = numerator / denominator
                            
                            numerator = np.sum(np.abs(test_y-oilForecast))
                            denominator = np.sum(np.abs(test_y))
                            wmape_f = numerator / denominator
                            
                            NQDS_h = compute_NQDS(trainY_main, oilPred_main)
                            NQDS_v = compute_NQDS(val_y, oilPred_validation)
                            NQDS_f = compute_NQDS(test_y, oilForecast)
                            
                            a, b = trainY_main, oilPred_main
                            relDiff_h = (b - a) / a * 100
                            relDiff_h [relDiff_h==np.inf] = 0
                            relDiff_h [relDiff_h==-np.inf] = 0
                            relDiff_h [np.isnan(relDiff_h)] = 0
                            
                            a, b = val_y, oilPred_validation
                            relDiff_v = (b - a) / a * 100
                            relDiff_v [relDiff_v==np.inf] = 0
                            relDiff_v [relDiff_v==-np.inf] = 0
                            relDiff_v [np.isnan(relDiff_v)] = 0
                            
                            a, b = test_y, oilForecast
                            relDiff_f = (b - a) / a * 100
                            relDiff_f [relDiff_f==np.inf] = 0
                            relDiff_f [relDiff_f==-np.inf] = 0
                            relDiff_f [np.isnan(relDiff_f)] = 0
                        except:
                            print('Model prediction is INF*************')
                            tf.keras.backend.clear_session()
                            
                            continue

                        
                        df_res = pd.concat([df_res, pd.DataFrame([reg,
                                                                  w,
                                                                  lg,
                                                                  prd,
                                                                  loss_short_name,
                                                                  round(mse_h.numpy(), 2),
                                                                  round(mse_v.numpy(), 2),
                                                                  round(mse_f.numpy(), 2),
                                                                  np.round(np.mean(relDiff_h), 2),
                                                                  np.round(np.mean(relDiff_v), 2),
                                                                  np.round(np.mean(relDiff_f), 2),
                                                                  np.round(NQDS_h, 2),
                                                                  np.round(NQDS_v, 2),
                                                                  np.round(NQDS_f, 2),
                                                                  round(mae_h, 2),
                                                                  round(mae_f, 2),
                                                                  round(mae_v, 2),
                                                                  round(wmape_h, 2),
                                                                  round(wmape_v, 2),
                                                                  round(wmape_f, 2),
                                                                  round(time_cost.total_seconds(), 2)
                                                                  # Processing Time
                                                                  ], df_result_column_names).T], ignore_index=True)
                        df_res.to_csv(path2main_output_folder + path2csvs + f'{atts_name}_dl_results.csv',
                                      index=False)
                        print("mse_h %2.2f, mse_f %2.2f, MAE_h %2.2f, MAE_v %2.2f,MAE_f %2.2f, WMAPE_h %2.2f, WMAPE_v %2.2f, WMAPE_f %2.2f" % (mse_h,mse_f, mae_h,mae_v,mae_f, wmape_h,wmape_v,wmape_f))
                        np.save(
                            path2main_output_folder + path2signal + f'{w}_{reg}_{lg}_{prd}_{loss_short_name}.npy',
                            predData)
                        



#Training deep-learning models
train_DL_models()

