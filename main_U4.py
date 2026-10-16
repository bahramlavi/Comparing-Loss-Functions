"""
This code implements the published paper in the journal of Scientific Reports,
entitled "Impact of Loss Function Selection on Deep Learning Methods for Oil and Water Production Forecasting in Offshore Reservoirs"
Code developed by Bahram Lavi and João Bertini.
paper DOI:
"""

from warnings import filterwarnings
filterwarnings('ignore')

import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"

# Disable automatic XLA clustering before importing TensorFlow
os.environ.pop("TF_XLA_FLAGS", None)
os.environ.pop("XLA_FLAGS", None)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from pathlib import Path
import argparse
import sys
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
print("GPUs:", gpus)

# GPU CONFIGURATION
tf.config.optimizer.set_jit(False)
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error)

from sklearn.preprocessing import MinMaxScaler

tf.keras.backend.set_floatx('float32')
tf.reduce_retracing = True
# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================
model_prediction = 'Qw'

# DS-I, DS-II, DS-III
scenario = 'DS-III'

# ============================================================
# PATHS
# ============================================================

path_history_data = './u4/'
path2main_output_folder = './Experiments/'
Path(path2main_output_folder).mkdir(parents=True, exist_ok=True)

path2signal = 'Signals/'
Path(path2main_output_folder + path2signal).mkdir(parents=True, exist_ok=True)

path2csvs = 'CSVs/'
Path(path2main_output_folder + path2csvs).mkdir(parents=True, exist_ok=True)

path2figures = 'Figures'
Path(path2main_output_folder + path2figures).mkdir(parents=True, exist_ok=True)


# ============================================================
# WELL DEFINITIONS
# ============================================================
PRODUCER_NAMES = [
    'P11',
    'P12',
    'P13',
    'P14',
    'P15',
    'P16'
]
GAS_INJECTOR_NAMES = [
    'I11G',
    'I12G',
    'I13G',
    'I14G',
    'I15G',
    'I16G',
    'P17G'

]
WATER_INJECTOR_NAMES = [
    'I11W',
    'I12W',
    'I13W',
    'I14W',
    'I15W',
    'P17W'
]

# ============================================================
# FORECAST HORIZONS
periods_h = [7, 30]

# ============================================================
# DATA INTERVAL
history_day_start = 0
history_day_end = -1

# 80% train / 10% validation / 10% test
n_trainable_data = 0.80
# ============================================================
# OUTPUT COLUMNS
df_result_column_names = [
    'Regressor',
    'Well',
    'Scenario',
    'lag',
    'period',
    'loss',
    'MSE_h',
    'MSE_v',
    'MSE_f',
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
    'MAPE_h',
    'MAPE_v',
    'MAPE_f',
    'Proc. Time',
    'Hyperparameters']

# ============================================================
# DATA FUNCTIONS
# ============================================================
def parse_portuguese_date(date_series):
    """
    Converts Portuguese month strings
    (e.g., '01-out-2021') into pandas datetime.
    """
    pt_to_en_months = {
        'jan': 'Jan',
        'fev': 'Feb',
        'mar': 'Mar',
        'abr': 'Apr',
        'mai': 'May',
        'jun': 'Jun',
        'jul': 'Jul',
        'ago': 'Aug',
        'set': 'Sep',
        'out': 'Oct',
        'nov': 'Nov',
        'dez': 'Dec'
    }

    date_str = (date_series.astype(str).str.lower())

    for pt, en in pt_to_en_months.items():
        date_str = date_str.str.replace(pt, en, regex=False)

    return pd.to_datetime(date_str, dayfirst=True, errors='coerce')


def read_csv_file(filepath):
    """
    Reads CSV with robust handling for
    encodings and delimiters.
    """
    if not os.path.exists(filepath):
        alt_path = filepath.lower()
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(
                f"Dataset file not found: {filepath}"
            )
    try:
        df = pd.read_csv(filepath, sep=';', encoding='utf-8-sig')
        if len(df.columns) <= 1:
            df = pd.read_csv(filepath, sep=',', encoding='utf-8-sig')
    except Exception:
        df = pd.read_csv(filepath, sep=',', encoding='latin1')
    df.columns = (df.columns.str.strip().str.replace('\ufeff', '', regex=False))

    return df

def parse_time_column(series):
    """
    Parses a temporal column.
    If all values can be interpreted as numeric, they are treated
    as simulation time. Otherwise, the original Portuguese date
    parser is used.
    """
    numeric = pd.to_numeric(series, errors='coerce')
    # Simulation time, e.g. 1200.0, 1201.0, ...
    if numeric.notna().all():
        return numeric.astype(float)
    # Calendar date
    return parse_portuguese_date(series)

def load_data(target_well_id):
    """
    Loads and merges UNISIM-IV dataset files
    according to the selected scenario.
    DS-I:
        Only the target well.
    DS-II:
        All producer wells.
    DS-III:
        Producers + gas injectors + water injectors.
    """

    # DS-I
    if scenario == 'DS-I':
        file_path = os.path.join(path_history_data, f'{target_well_id}.csv')
        df = read_csv_file(file_path)
        # Identify temporal column
        date_col = [c for c in df.columns if any(k in c.lower() for k in ['date', 'dia', 'time'])]

        if date_col:
            original_date_col = date_col[0]

            df['Date'] = parse_time_column(
                df[original_date_col]
            )

            if original_date_col != 'Date':
                df = df.drop(columns=[original_date_col])

            if df['Date'].isna().any():
                raise ValueError(
                    f"NaN values found in Date "
                    f"for well {target_well_id}"
                )

            if not df['Date'].is_unique:
                duplicated = (
                    df.loc[
                        df['Date'].duplicated(
                            keep=False
                        ),
                        'Date'
                    ]
                    .value_counts()
                    .head(10)
                )

                raise ValueError(
                    f"Duplicated Date values found "
                    f"for well {target_well_id}:\n"
                    f"{duplicated}"
                )

        return (df.sort_values('Date').reset_index(drop=True))


    # DS-II
    elif scenario == 'DS-II':
        merged_df = None
        for p_name in PRODUCER_NAMES:
            file_path = os.path.join(path_history_data, f'{p_name}.csv')
            p_df = read_csv_file(file_path)
            # Identify temporal column
            date_col = [c for c in p_df.columns if any(k in c.lower() for k in ['date', 'dia', 'time'])]
            if not date_col:
                raise ValueError(
                    f"No Date/Time column found "
                    f"in file {p_name}.csv"
                )
            original_date_col = date_col[0]

            # Parse temporal column
            p_df['Date'] = parse_time_column(
                p_df[original_date_col]
            )
            if original_date_col != 'Date':
                p_df = p_df.drop(columns=[original_date_col])

            if p_df['Date'].isna().any():
                raise ValueError(
                    f"NaN values found in Date "
                    f"for producer {p_name}"
                )
            if not p_df['Date'].is_unique:
                duplicated = (
                    p_df.loc[
                        p_df['Date'].duplicated(
                            keep=False
                        ),
                        'Date'
                    ]
                    .value_counts()
                    .head(10)
                )
                raise ValueError(
                    f"Duplicated Date values found "
                    f"for producer {p_name}:\n"
                    f"{duplicated}"
                )

            rename_dict = {col: f'{p_name}_{col}' for col in p_df.columns if col != 'Date'}

            p_df = p_df.rename(columns=rename_dict)

            # Merge
            if merged_df is None:
                merged_df = p_df.copy()
            else:
                merged_df = pd.merge(merged_df, p_df, on='Date', how='outer', validate='one_to_one')

        if merged_df is None:
            raise ValueError("No producer data were loaded.")
        return (merged_df.sort_values('Date').reset_index(drop=True).fillna(0))


    # DS-III
    elif scenario == 'DS-III':
        merged_df = None
        all_wells = (PRODUCER_NAMES + GAS_INJECTOR_NAMES + WATER_INJECTOR_NAMES)

        for w_name in all_wells:
            file_path = os.path.join(path_history_data, f'{w_name}.csv')
            try:
                w_df = read_csv_file(file_path)
            except FileNotFoundError:
                print(
                    f"Warning: file not found: "
                    f"{w_name}.csv"
                )
                continue


            # Identify temporal column
            date_col = [c for c in w_df.columns if any(k in c.lower() for k in ['date', 'dia', 'time'])]
            if not date_col:
                raise ValueError(
                    f"No Date/Time column found "
                    f"in file {w_name}.csv"
                )
            original_date_col = date_col[0]

            # Parse temporal column
            w_df['Date'] = parse_time_column(
                w_df[original_date_col]
            )
            if original_date_col != 'Date':
                w_df = w_df.drop(columns=[original_date_col])

            if w_df['Date'].isna().any():
                raise ValueError(f"NaN values found in Date "
                                 f"for well {w_name}")
            if not w_df['Date'].is_unique:
                duplicated = (
                    w_df.loc[
                        w_df['Date'].duplicated(
                            keep=False
                        ),
                        'Date'
                    ]
                    .value_counts()
                    .head(10)
                )
                raise ValueError(
                    f"Duplicated Date values found "
                    f"for well {w_name}:\n"
                    f"{duplicated}"
                )

            rename_dict = {
                col: f'{w_name}_{col}'
                for col in w_df.columns
                if col != 'Date'
            }
            w_df = w_df.rename(
                columns=rename_dict
            )

            # Merge

            if merged_df is None:
                merged_df = w_df.copy()
            else:
                merged_df = pd.merge(merged_df, w_df, on='Date', how='outer', validate='one_to_one')
        if merged_df is None:
            raise ValueError("No well data were loaded.")
        return (merged_df.sort_values('Date').reset_index(drop=True).fillna(0))


    else:
        raise ValueError(f"Unknown scenario: {scenario}")
# ============================================================
# DIRECT h-STEP-AHEAD TARGET
# ============================================================
def add_prediction_target(
        data,
        prediction_var,
        dah=1):
    """
    Creates the direct h-step-ahead target.
    At time t:
        pred(t) = prediction_var(t + dah)
    """
    if prediction_var not in data.columns:
        raise ValueError(
            f"Prediction variable "
            f"'{prediction_var}' not found. "
            f"Available columns: "
            f"{list(data.columns)}")

    df = data.copy()
    df['pred'] = (df[prediction_var].shift(-dah))
    # Last dah observations have no future target
    df.dropna(subset=['pred'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
# ============================================================
# ZERO HANDLING
# ============================================================
def remove_zero_values(
        data,
        target_col='pred'
):
    """
    Replaces internal sequences of zeros in the
    target with the average of the immediately
    preceding and following values.
    """
    if target_col not in data.columns:

        raise ValueError(
            f"Target column "
            f"'{target_col}' not found. "
            f"Available columns: "
            f"{list(data.columns)}")

    df = data.copy()
    zero_indices = df.index[
        df[target_col] == 0
        ].to_numpy()

    if len(zero_indices) == 0:
        return df
    zero_sequences = np.split(
        zero_indices,
        np.where(
            np.diff(zero_indices) != 1
        )[0] + 1
    )
    for sequence in zero_sequences:
        if len(sequence) == 0:
            continue
        first_index = sequence[0]
        last_index = sequence[-1]
        if (first_index > 0 and last_index < len(df) - 1):
            previous_value = df.loc[
                first_index - 1,
                target_col
            ]
            next_value = df.loc[
                last_index + 1,
                target_col
            ]
            replacement_value = (previous_value + next_value) / 2.0

            df.loc[sequence, target_col] = replacement_value

    return df


# ============================================================
# METRICS
# ============================================================
def compute_NQDS(y, y_hat):

    tolerance = 2
    cons = 20

    AQD = np.sum((tolerance * y + cons) ** 2)
    SD = np.sum(y_hat - y)
    if SD != 0:
        QDS = ((SD / np.abs(SD)) * np.sum(y_hat - y) ** 2)
    else:
        QDS = (np.sum(y_hat - y) ** 2)
    NQDS = QDS / AQD

    return np.round(NQDS, 2)

def calculate_mse(
        y_true,
        y_pred
):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def calculate_mape(y_true, y_pred):
    return (mean_absolute_percentage_error(y_true, y_pred))


def compute_reldiff(actual, predicted):
    actual = np.asarray(actual, dtype=np.float64)

    predicted = np.asarray(predicted, dtype=np.float64)

    with np.errstate(divide='ignore', invalid='ignore'):
        diff = ((predicted - actual) / actual * 100)

    diff[~np.isfinite(diff)] = 0

    return np.mean(diff)


# ============================================================
# TRAIN DD MODELS
# ============================================================
def train_DL_models():
    from DL_Regressors import call_DL_Regressor
    from Losses import call_Losses
    # PRODUCERS
    for well_id in PRODUCER_NAMES:

        df_res = pd.DataFrame(columns=df_result_column_names)

        results_csv_path = (
                path2main_output_folder + path2csvs + f'{well_id}_'
                                                      f'Scenario{scenario}_'
                                                      f'{model_prediction}_results.csv')

        if os.path.exists(results_csv_path):
            try:
                df_res = pd.read_csv(results_csv_path)
            except Exception:
                pass

        print(f'\n{"=" * 60}\n'
              f'Processing Sepia Dataset | '
              f'Well = {well_id} | '
              f'Scenario = {scenario} | '
              f'Target = {model_prediction}'
              f'\n{"=" * 60}\n')


        # FORECAST HORIZON
        for prd in periods_h:
            # LOAD DATA
            df = load_data(well_id)

            # DEFINE TARGET AND INPUT ATTRIBUTES
            if scenario == 'DS-I':
                target_col = (f'{model_prediction}' if model_prediction in df.columns else [c for c in df.columns if
                    model_prediction in c])

                if isinstance(target_col, list):
                    target_col = (target_col[0])

                attribute_names = [col for col in df.columns if col.lower() not in ['date', 'dia', 'time', 'pred']]
                prediction_variable = (target_col)
            elif scenario == 'DS-II':
                prediction_variable = (f'{well_id}_'
                                       f'{model_prediction}')
                attribute_names = [col for col in df.columns if col.lower() not in ['date', 'dia', 'time', 'pred']]
            else:
                if (f'{well_id}_{model_prediction}' in df.columns):
                    prediction_variable = (f'{well_id}_'
                                           f'{model_prediction}')

                else:
                    prediction_variable = (f'{well_id}_Qg')
                attribute_names = [col for col in df.columns if col.lower() not in ['date', 'dia', 'time', 'pred']]

            # SELECT EMPLOYED HISTORICAL INTERVAL
            if (history_day_end is None or history_day_end == -1):
                df = df.iloc[history_day_start:].copy()
            else:
                df = df.iloc[history_day_start:
                             history_day_end].copy()

            df.reset_index(drop=True, inplace=True)
            # DIRECT h-DAY TARGET
            df_main = add_prediction_target(data=df[attribute_names].copy(), prediction_var=(prediction_variable),
                dah=prd)
            # HANDLE NON-FINITE INPUT VALUES
            df_main[attribute_names] = (df_main[attribute_names].replace([np.inf, -np.inf], np.nan))

            missing_values = (df_main[attribute_names].isna().sum())
            missing_values = (missing_values[missing_values > 0])
            if not missing_values.empty:
                print(

                    f'[{well_id}] '
                    f'Missing input values '
                    f'before treatment:')
                print(missing_values)
            # FORWARD FILL ONLY
            df_main[attribute_names] = (
                df_main[attribute_names].ffill())

            # Initial values that cannot be forward-filled
            df_main.dropna(subset=(attribute_names + ['pred']), inplace=True)
            df_main.reset_index(drop=True, inplace=True)

            # =================================================
            # CHRONOLOGICAL 80 / 10 / 10 SPLIT
            #
            # IMPORTANT:
            # Split occurs BEFORE temporal-window generation.
            # Lag is subsequently selected by Random Search.
            # =================================================
            n_samples = len(df_main)
            split_index = int(n_samples * n_trainable_data)
            remaining_samples = (n_samples - split_index)
            val_len = int(remaining_samples * 0.5)
            train_df = df_main.iloc[:split_index].copy()

            val_df = df_main.iloc[split_index:
                                  split_index + val_len].copy()

            test_df = df_main.iloc[split_index + val_len:].copy()

            if (len(train_df) == 0 or len(val_df) == 0 or len(test_df) == 0):
                print(
                    f'[{well_id}] '
                    f'Insufficient samples '
                    f'for horizon={prd}. '
                    f'Skipping.')

                continue


            # RAW 2D INPUT DATA
            X_train_raw = (train_df[attribute_names].to_numpy(dtype=np.float32))
            X_val_raw = (val_df[attribute_names].to_numpy(dtype=np.float32))
            X_test_raw = (test_df[attribute_names].to_numpy(dtype=np.float32))

            # RAW TARGETS
            y_train_raw = (train_df['pred'].to_numpy(dtype=np.float32))
            y_val_raw = (val_df['pred'].to_numpy(dtype=np.float32))
            y_test_raw = (test_df['pred'].to_numpy(dtype=np.float32))

            # INPUT SCALER
            # Fit exclusively on training.
            scaler = MinMaxScaler(feature_range=(0, 1))
            X_train_scaled = (scaler.fit_transform(X_train_raw).astype(np.float32))
            X_val_scaled = (scaler.transform(X_val_raw).astype(np.float32))
            X_test_scaled = (scaler.transform(X_test_raw).astype(np.float32))


            # TARGET SCALER
            # Fit exclusively on training.
            scalerLA = MinMaxScaler(feature_range=(0, 1))
            y_train_scaled = (scalerLA.fit_transform(y_train_raw.reshape(-1, 1)).reshape(-1).astype(np.float32))
            y_val_scaled = (scalerLA.transform(y_val_raw.reshape(-1, 1)).reshape(-1).astype(np.float32))
            y_test_scaled = (scalerLA.transform(y_test_raw.reshape(-1, 1)).reshape(-1).astype(np.float32))

            # DIAGNOSTICS
            print(
                f'\n[{well_id}] '
                f'Horizon={prd}'
            )


            print(
                'Raw/scaled split shapes:',
                f'train={X_train_scaled.shape}',
                f'val={X_val_scaled.shape}',
                f'test={X_test_scaled.shape}'
            )

            print('trainX =', np.isnan(X_train_scaled).sum(), 'trainY =', np.isnan(y_train_scaled).sum(),
                'valX =', np.isnan(X_val_scaled).sum(), 'valY =', np.isnan(y_val_scaled).sum(), 'testX =',
                np.isnan(X_test_scaled).sum(), 'testY =', np.isnan(y_test_scaled).sum())

            # LOSS CONFIGURATIONS
            loss_configurations = (call_Losses())
            for config in loss_configurations:

                loss_short_name = (config['short_name'])

                loss_name = (config['name'])
                loss_fn = (config['function'])
                # DD ARCHITECTURES
                regs = (call_DL_Regressor())
                for reg in regs.keys():
                    sig_filename = (f'{model_prediction}_'
                                    f'{well_id}_'
                                    f'Scen{scenario}_'
                                    f'{reg}_'
                                    f'{prd}_'
                                    f'{loss_short_name}.npy')

                    sig_path = (path2main_output_folder + path2signal + sig_filename)
                    if os.path.isfile(sig_path):
                        continue

                    print(f'\n[{well_id}] '
                          f'Regressor={reg}, '
                          f'Horizon={prd}, '
                          f'Loss={loss_short_name}'
                          )
                    # =========================================
                    # RANDOM SEARCH
                    # Internally optimizes:
                    # - lag
                    # - learning rate
                    # - batch size
                    # - dropout
                    # - architecture-specific parameters
                    # =========================================
                    (oilPred_main, oilForecast, time_cost, reg_f) = regs[reg](
                        X_train_scaled,
                        X_test_scaled,
                        y_train_scaled,
                        X_val_scaled,
                        y_val_scaled,
                        y_test_scaled,
                        loss_fn
                    )


                    # BEST LAG
                    best_lag = int(reg_f.best_lag_)
                    print(f'[{well_id}] '
                          f'Best lag={best_lag}')
                    print('Best hyperparameters:', reg_f.best_params_)
                    print('Best validation MAE '
                          '(scaled data):', reg_f.best_val_mae_)

                    # VALIDATION PREDICTION
                    oilPred_validation = (reg_f.predict(reg_f.validation_X_, verbose=0))

                    # INVERSE TRANSFORM PREDICTIONS
                    oilPred_main = (scalerLA.inverse_transform(oilPred_main.reshape(-1, 1)))
                    oilPred_validation = (scalerLA.inverse_transform(oilPred_validation.reshape(-1, 1)))
                    oilForecast = (scalerLA.inverse_transform(oilForecast.reshape(-1, 1)))

                    # INVERSE TRANSFORM ALIGNED TARGETS
                    trainY_main = (scalerLA.inverse_transform(reg_f.train_y_.reshape(-1, 1)))
                    val_y_tmp = (scalerLA.inverse_transform(reg_f.validation_y_.reshape(-1, 1)))
                    test_y_tmp = (scalerLA.inverse_transform(reg_f.test_y_.reshape(-1, 1)))

                    # COMPLETE SIGNALS
                    predData = np.vstack((oilPred_main,oilPred_validation,oilForecast))
                    actualData = np.vstack(
                        (
                            trainY_main,
                            val_y_tmp,
                            test_y_tmp
                        )
                    )
                    # MSE
                    mse_h = calculate_mse(trainY_main, oilPred_main)
                    mse_v = calculate_mse(val_y_tmp, oilPred_validation)
                    mse_f = calculate_mse(test_y_tmp, oilForecast)

                    # MAE
                    mae_h = mean_absolute_error(trainY_main, oilPred_main)
                    mae_v = mean_absolute_error(val_y_tmp, oilPred_validation)
                    mae_f = mean_absolute_error(test_y_tmp, oilForecast)

                    # WMAPE
                    denominator_h = np.sum(np.abs(trainY_main))
                    wmape_h = (
                        np.sum(np.abs(trainY_main - oilPred_main)) / denominator_h
                        if denominator_h != 0
                        else 0)
                    denominator_v = np.sum(np.abs(val_y_tmp))
                    wmape_v = (
                        np.sum(np.abs(val_y_tmp - oilPred_validation)) / denominator_v
                        if denominator_v != 0
                        else 0)
                    denominator_f = np.sum(np.abs(test_y_tmp))
                    wmape_f = (
                        np.sum(np.abs(test_y_tmp - oilForecast)) / denominator_f
                        if denominator_f != 0
                        else 0)

                    # MAPE
                    mape_h = calculate_mape(trainY_main, oilPred_main)
                    mape_v = calculate_mape(val_y_tmp, oilPred_validation)
                    mape_f = calculate_mape(test_y_tmp, oilForecast)

                    # NQDS
                    NQDS_h = compute_NQDS(trainY_main, oilPred_main)
                    NQDS_v = compute_NQDS(val_y_tmp, oilPred_validation)
                    NQDS_f = compute_NQDS(test_y_tmp, oilForecast)

                    # RELATIVE DIFFERENCE
                    relDiff_h = compute_reldiff(trainY_main, oilPred_main)
                    relDiff_v = compute_reldiff(val_y_tmp, oilPred_validation)
                    relDiff_f = compute_reldiff(test_y_tmp, oilForecast)

                    # SAVE RESULT ROW
                    new_row = pd.DataFrame(
                        [{'Regressor': reg, 'Well': well_id, 'Scenario': scenario, 'lag': best_lag, 'period': prd,
                            'loss': loss_short_name, 'MSE_h': round(float(mse_h.numpy()), 2),
                            'MSE_v': round(float(mse_v.numpy()), 2), 'MSE_f': round(float(mse_f.numpy()), 2),
                            'relDiff_h': np.round(relDiff_h, 2), 'relDiff_v': np.round(relDiff_v, 2),
                            'relDiff_f': np.round(relDiff_f, 2), 'NQDS_h': np.round(NQDS_h, 2),
                            'NQDS_v': np.round(NQDS_v, 2), 'NQDS_f': np.round(NQDS_f, 2),
                            'MAE_h': round(float(mae_h), 2), 'MAE_v': round(float(mae_v), 2),
                            'MAE_f': round(float(mae_f), 2), 'WMAPE_h': round(float(wmape_h), 2),
                            'WMAPE_v': round(float(wmape_v), 2), 'WMAPE_f': round(float(wmape_f), 2),
                            'MAPE_h': round(float(mape_h), 2), 'MAPE_v': round(float(mape_v), 2),
                            'MAPE_f': round(float(mape_f), 2), 'Proc. Time': round(time_cost.total_seconds(), 2),
                            'Hyperparameters': str(reg_f.best_params_)}])

                    df_res = pd.concat([df_res, new_row], ignore_index=True)

                    df_res.to_csv(results_csv_path, index=False)
                    # SAVE SIGNALS
                    np.save(sig_path, [actualData, predData])

                    tf.keras.backend.clear_session()


# ============================================================
# GENERATE RESULTS FOR BEST MODELS
# ============================================================
def generate_results_best_models():
    for well_id in PRODUCER_NAMES:
        results_csv_path = (
                path2main_output_folder
                + path2csvs
                + f'{well_id}_'
                  f'Scenario{scenario}_'
                  f'{model_prediction}_results.csv'
        )
        if not os.path.exists(
                results_csv_path
        ):
            continue
        df = pd.read_csv(results_csv_path)
        if df.empty:
            continue

        # BEST VALIDATION MODEL PER FORECAST HORIZON
        df_res = df.loc[df.groupby(by=['period'], sort=True)['MAE_v'].idxmin()]
        best_csv_filename = (
                path2main_output_folder
                + f'{model_prediction}_'
                  f'{well_id}_'
                  f'Scenario{scenario}_'
                  f'bestmodels.csv'
        )
        df_res.to_csv(
            best_csv_filename,
            index=False
        )

        # PLOTS
        for i in range(len(df_res)):
            regressor = (df_res.iloc[i].Regressor)
            lag = int(df_res.iloc[i].lag)
            period = (df_res.iloc[i].period)
            loss = (df_res.iloc[i].loss)

            # Lag is no longer part of filename
            sig_filename = (
                f'{model_prediction}_'
                f'{well_id}_'
                f'Scen{scenario}_'
                f'{regressor}_'
                f'{period}_'
                f'{loss}.npy'
            )

            sig_path = (path2main_output_folder + path2signal + sig_filename)

            if not os.path.exists(sig_path):
                continue

            data = np.load(sig_path, allow_pickle=True)
            actual = data[0]
            pred = data[1]

            font = {'weight': 'bold', 'size': 11}
            plt.rc('font', **font)

            plt.figure(figsize=(7, 3))
            plt.plot(actual, 'bo', markersize=3.5, label='Actual')
            plt.plot(pred, 'ro', markersize=3.5, label='Pred')
            plt.ylabel(
                f'{well_id}, '
                f'{model_prediction}\n'
                f'[Rate]',
                fontweight='bold')

            plt.xlabel('Time (days)', fontweight='bold')
            plt.legend()
            plt.grid(axis='both')
            plt.title(
                f'{period}-Day forecast, '
                f'{regressor} on {loss}, '
                f'L={lag} '
                f'(Scen {scenario})',
                fontsize=12,
                fontweight='bold')

            fig_path = (
                f'{path2main_output_folder}'
                f'{path2figures}/'
                f'{model_prediction}_'
                f'{well_id}_'
                f'Scen{scenario}_'
                f'{regressor}_'
                f'{period}.png')

            plt.savefig(fig_path, dpi=200, bbox_inches='tight')

            plt.close()


if __name__ == '__main__':
    print("Python:", sys.executable)
    print("TensorFlow:", tf.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))

    train_DL_models()
    generate_results_best_models()

