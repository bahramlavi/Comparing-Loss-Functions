"""
Code by Bahram Lavi and João Bertini
Description: This code implements deep learning models and their associated hyperparameters.
"""

import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision

mixed_precision.set_global_policy('mixed_float16')

from scipy.stats import loguniform, uniform
from sklearn.model_selection import ParameterSampler

from tensorflow.keras.layers import (
    Bidirectional,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    LSTM,
    MaxPooling1D,
)

from tcn import TCN


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ITER = 20

MAX_EPOCHS = 300
PATIENCE = 12

UNITS = [16, 32, 64, 128]

BATCH_SIZES = [32, 64, 128, 256]

LAG_WINDOWS =  [7, 14, 30, 60, 90]  # [2, 4, 6, 8]

FILTERS = [16, 32, 64, 128]

CNN_KERNELS_1 = [2, 3, 5, 7]
CNN_KERNELS_2 = [2, 3, 5]

TCN_KERNELS = [2, 3, 5, 7]

TCN_DILATIONS = [
    (1, 2, 4),
    (1, 2, 4, 8),
    (1, 2, 4, 8, 16),
]

DENSE_UNITS = [16, 32, 64]
ACTIVATION = "tanh"


# DEVICE
available_gpus = tf.config.list_physical_devices("GPU")
device = "/GPU:0" if available_gpus else "/CPU:0"
print(f"TensorFlow device: {device}")


# REPRODUCIBILITY
def set_seed(seed):
    tf.keras.utils.set_random_seed(seed)

# ============================================================
# OPTIMIZER
# ============================================================
def new_adam_optimizer(learning_rate):

    return tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=3.0,
    )

# ============================================================
# CALLBACKS
# ============================================================
def get_callbacks():
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_mae",
            mode="min",
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=0,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


# ============================================================
# SEQUENCE GENERATION
# ============================================================
def create_sequences(X, y, lag):
    X_seq = []
    y_seq = []
    for i in range(lag, len(X)):
        X_seq.append(X[i - lag:i])
        y_seq.append(y[i])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
    )


# ============================================================
# VALIDATION SEQUENCES
# ============================================================
def create_validation_sequences(
        X_train,
        y_train,
        X_val,
        y_val,
        lag,
):
    """
    Builds validation sequences while preserving the historical
    observations immediately preceding the validation period.

    No future information is introduced.
    """

    X_context = np.concatenate([X_train[-lag:], X_val, ], axis=0, )
    y_context = np.concatenate([y_train[-lag:], y_val, ], axis=0, )
    X_seq = []
    y_seq = []
    for i in range(lag, len(X_context)):
        X_seq.append(X_context[i - lag:i])

        y_seq.append(y_context[i])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
    )


# ============================================================
# TEST SEQUENCES
# ============================================================

def create_test_sequences(
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        lag,
):
    """
    Builds test sequences using only observations available
    before each target time.

    The context preceding the test period comes from the end
    of the validation period.
    """

    X_history = np.concatenate([X_train, X_val, ], axis=0, )

    y_history = np.concatenate([y_train, y_val, ], axis=0, )

    X_context = np.concatenate([X_history[-lag:], X_test, ], axis=0, )

    y_context = np.concatenate([y_history[-lag:], y_test, ], axis=0, )

    X_seq = []
    y_seq = []

    for i in range(lag, len(X_context)):
        X_seq.append(X_context[i - lag:i])

        y_seq.append(y_context[i])

    return (
        np.asarray(X_seq, dtype=np.float32),
        np.asarray(y_seq, dtype=np.float32),
    )


# ============================================================
# RANDOM SEARCH INCLUDING LAG WINDOW
# ============================================================

def random_search(
        build_model,
        param_distributions,
        X_train_raw,
        y_train_raw,
        X_val_raw,
        y_val_raw,
        X_test_raw,
        y_test_raw,
        loss,
        n_iter=N_ITER,
):

    sampled_parameters = list(

        ParameterSampler(
            param_distributions=param_distributions,
            n_iter=n_iter,
            random_state=RANDOM_STATE,
        )
    )

    best_score = np.inf

    best_params = None
    best_seed = None

    search_history = []

    # RANDOM SEARCH
    for trial, params in enumerate(
            sampled_parameters,
            start=1,
    ):

        trial_seed = RANDOM_STATE + trial

        set_seed(trial_seed)

        tf.keras.backend.clear_session()

        lag = params["lag"]

        batch_size = params["batch_size"]

        # Recreate sequences for this lag
        train_X, train_y = create_sequences(
            X_train_raw,
            y_train_raw,
            lag,
        )

        X_val, y_val = create_validation_sequences(
            X_train_raw,
            y_train_raw,
            X_val_raw,
            y_val_raw,
            lag,
        )

        # Model-specific parameters
        model_params = {

            key: value

            for key, value in params.items()

            if key not in [
                "lag",
                "batch_size",
            ]
        }

        # Build model
        with tf.device(device):

            model = build_model(
                input_shape=(
                    train_X.shape[1],
                    train_X.shape[2],
                ),
                loss=loss,
                **model_params,
            )

            history = model.fit(
                train_X,
                train_y,
                validation_data=(
                    X_val,
                    y_val,
                ),
                epochs=MAX_EPOCHS,
                batch_size=batch_size,
                callbacks=get_callbacks(),
                shuffle=False,
                verbose=0,
            )

        # Validation MAE
        val_pred = model.predict(
            X_val,
            verbose=0,
        ).reshape(-1)

        y_val_flat = y_val.reshape(-1)

        val_mae = np.mean(
            np.abs(
                y_val_flat - val_pred
            )
        )

        epochs_used = len(
            history.history["loss"]
        )

        search_history.append(
            {
                "trial": trial,
                "val_mae": float(val_mae),
                "epochs": epochs_used,
                **params,
            }
        )

        # print(
        #     f"Trial {trial:02d}/{n_iter} | "
        #     f"Lag = {lag:3d} | "
        #     f"Val MAE = {val_mae:.6f} | "
        #     f"Epochs = {epochs_used}"
        # )
        # ========================================================
        # BEST CONFIGURATION - NO RETRAIN
        # ========================================================

        if val_mae < best_score:

            best_score = val_mae
            best_params = params.copy()
            best_seed = trial_seed

            # With restore_best_weights=True, these correspond
            # to the best validation epoch of this trial.
            best_weights = [
                w.copy()
                for w in model.get_weights()
            ]

            best_lag = lag

            best_train_X = train_X
            best_train_y = train_y

            best_val_X = X_val
            best_val_y = y_val

        # release graph/model before next trial
        del model
        tf.keras.backend.clear_session()

        print("\nBest hyperparameters:")

        for key, value in best_params.items():
            print(f"{key}: {value}")

        print(
            f"Best validation MAE: "
            f"{best_score:.6f}"
        )


        # ========================================================
        # DATA ASSOCIATED WITH BEST TRIAL
        # ========================================================

        train_X = best_train_X
        train_y = best_train_y

        X_val = best_val_X
        y_val = best_val_y

        best_lag = best_lag


        # ========================================================
        # CREATE TEST DATA USING THE SELECTED LAG
        # ========================================================

        test_X, test_y = create_test_sequences(
            X_train_raw,
            y_train_raw,
            X_val_raw,
            y_val_raw,
            X_test_raw,
            y_test_raw,
            best_lag,
        )
        # ========================================================
        # REBUILD SELECTED MODEL
        # ========================================================

        set_seed(best_seed)

        tf.keras.backend.clear_session()

        best_model_params = {
            key: value
            for key, value in best_params.items()
            if key not in [
                "lag",
                "batch_size",
            ]
        }

        with tf.device(device):

            best_model = build_model(
                input_shape=(
                    best_train_X.shape[1],
                    best_train_X.shape[2]
                ),
                loss=loss,
                **best_model_params
            )

            # Restore exactly the weights of the winning trial
            best_model.set_weights(
                best_weights
            )


        # PREDICTIONS
        train_pred = best_model.predict(train_X, verbose=0, )
        test_pred = best_model.predict(test_X, verbose=0, )


        # STORE SEARCH INFORMATION
        best_model.best_params_ = best_params
        best_model.best_val_mae_ = best_score
        best_model.best_lag_ = best_lag
        best_model.search_history_ = search_history


        # STORE DATA ASSOCIATED WITH SELECTED LAG
        best_model.train_X_ = train_X
        best_model.train_y_ = train_y

        best_model.validation_X_ = X_val
        best_model.validation_y_ = y_val

        best_model.test_X_ = test_X
        best_model.test_y_ = test_y


        return (
            train_pred,
            test_pred,
            test_y,
            best_model,
        )


# ============================================================
# REGRESSORS
# ============================================================

def call_DL_Regressor():
    # LSTM
    def get_gridsearch_with_LSTM(X_train, X_test, y_train, X_val, y_val, y_test, loss, ):
        def create_model(input_shape, units, dropout, n_layers, learning_rate, loss, ):
            model = tf.keras.Sequential()
            if n_layers == 1:
                model.add(LSTM(units, activation=ACTIVATION, dropout=dropout, input_shape=input_shape, ))
            else:
                model.add(LSTM(units, activation=ACTIVATION, dropout=dropout, return_sequences=True,
                    input_shape=input_shape, ))

                model.add(LSTM(units, activation=ACTIVATION, dropout=dropout, ))

            model.add(Dense(1, dtype='float32'))

            model.compile(optimizer=new_adam_optimizer(learning_rate), loss=loss,
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")], jit_compile=False, )

            return model

        param_distributions = {"units": UNITS, "dropout": uniform(0.0, 0.5, ), "n_layers": [1, 2],
            "learning_rate": loguniform(1e-4, 1e-2, ), "batch_size": BATCH_SIZES, "lag": LAG_WINDOWS, }
        start = datetime.datetime.now()
        (
            oilPred,
            oilForecast,
            test_y,
            reg_f,
        ) = random_search(
            create_model,
            param_distributions,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            loss,
        )

        end = datetime.datetime.now()

        return (
            oilPred,
            oilForecast,
            end - start,
            reg_f,
        )

    # ========================================================
    # BiLSTM
    # ========================================================
    def get_gridsearch_with_LSTMbidirectional(
            X_train,
            X_test,
            y_train,
            X_val,
            y_val,
            y_test,
            loss,
    ):

        def create_model(
                input_shape,
                units,
                dropout,
                n_layers,
                learning_rate,
                loss,
        ):

            model = tf.keras.Sequential()

            if n_layers == 1:

                model.add(
                    Bidirectional(
                        LSTM(
                            units,
                            activation=ACTIVATION,
                            dropout=dropout,
                        ),
                        input_shape=input_shape,
                    )
                )

            else:

                model.add(
                    Bidirectional(
                        LSTM(
                            units,
                            activation=ACTIVATION,
                            dropout=dropout,
                            return_sequences=True,
                        ),
                        input_shape=input_shape,
                    )
                )

                model.add(
                    Bidirectional(
                        LSTM(
                            units,
                            activation=ACTIVATION,
                            dropout=dropout,
                        )
                    )
                )

            model.add(
                Dense(1,dtype='float32')
            )

            model.compile(
                optimizer=new_adam_optimizer(
                    learning_rate
                ),
                loss=loss,
                metrics=[
                    tf.keras.metrics.MeanAbsoluteError(
                        name="mae"
                    )
                ],
                jit_compile=False,
            )

            return model

        param_distributions = {

            "units":
                UNITS,

            "dropout":
                uniform(
                    0.0,
                    0.5,
                ),

            "n_layers":
                [1, 2],

            "learning_rate":
                loguniform(
                    1e-4,
                    1e-2,
                ),

            "batch_size":
                BATCH_SIZES,

            "lag":
                LAG_WINDOWS,
        }

        start = datetime.datetime.now()

        (
            oilPred,
            oilForecast,
            test_y,
            reg_f,
        ) = random_search(
            create_model,
            param_distributions,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            loss,
        )

        end = datetime.datetime.now()

        return (
            oilPred,
            oilForecast,
            end - start,
            reg_f,
        )

    # ========================================================
    # CNN
    # ========================================================
    def get_gridsearch_with_CNN1D(
            X_train,
            X_test,
            y_train,
            X_val,
            y_val,
            y_test,
            loss,
    ):

        def create_model(
                input_shape,
                filters1,
                kernel1,
                filters2,
                kernel2,
                dense_units,
                dropout,
                learning_rate,
                loss,
        ):

            model = tf.keras.Sequential()

            model.add(
                Conv1D(
                    filters=filters1,
                    kernel_size=kernel1,
                    activation=ACTIVATION,
                    padding="same",
                    input_shape=input_shape,
                )
            )

            model.add(
                Conv1D(
                    filters=filters2,
                    kernel_size=kernel2,
                    activation=ACTIVATION,
                    padding="same",
                )
            )

            model.add(
                Dropout(dropout)
            )

            model.add(
                MaxPooling1D(
                    pool_size=2,
                    padding="same",
                )
            )

            model.add(
                Flatten()
            )

            model.add(
                Dense(
                    dense_units,
                    activation=ACTIVATION,
                )
            )

            model.add(
                Dense(1,dtype='float32')
            )

            model.compile(
                optimizer=new_adam_optimizer(
                    learning_rate
                ),
                loss=loss,
                metrics=[
                    tf.keras.metrics.MeanAbsoluteError(
                        name="mae"
                    )
                ],
                jit_compile=False,
            )

            return model

        param_distributions = {

            "filters1":
                FILTERS,

            "kernel1":
                CNN_KERNELS_1,

            "filters2":
                FILTERS,

            "kernel2":
                CNN_KERNELS_2,

            "dense_units":
                DENSE_UNITS,

            "dropout":
                uniform(
                    0.0,
                    0.5,
                ),

            "learning_rate":
                loguniform(
                    1e-4,
                    1e-2,
                ),

            "batch_size":
                BATCH_SIZES,

            "lag":
                LAG_WINDOWS,
        }

        start = datetime.datetime.now()

        (
            oilPred,
            oilForecast,
            test_y,
            reg_f,
        ) = random_search(
            create_model,
            param_distributions,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            loss,
        )

        end = datetime.datetime.now()

        return (
            oilPred,
            oilForecast,
            end - start,
            reg_f,
        )


    # ========================================================
    # TCN
    # ========================================================
    def get_gridsearch_with_TCN(X_train, X_test, y_train, X_val, y_val, y_test, loss, ):

        def create_model(input_shape, filters, kernel_size, dilations, dropout, learning_rate, loss, ):
            model = tf.keras.Sequential()

            model.add(TCN(nb_filters=filters, kernel_size=kernel_size, dilations=dilations, dropout_rate=dropout,
                return_sequences=False, input_shape=input_shape, ))

            model.add(Dense(1, dtype='float32'))

            model.compile(optimizer=new_adam_optimizer(learning_rate), loss=loss,
                metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")], jit_compile=False, )

            return model

        param_distributions = {
            "filters": FILTERS,
            "kernel_size": TCN_KERNELS,
            "dilations": TCN_DILATIONS,
            "dropout": uniform(0.0, 0.5, ),
            "learning_rate": loguniform(1e-4, 1e-2, ),
            "batch_size": BATCH_SIZES,
            "lag": LAG_WINDOWS, }

        start = datetime.datetime.now()
        (
            oilPred,
            oilForecast,
            test_y,
            reg_f,
        ) = random_search(
            create_model,
            param_distributions,
            X_train,
            y_train,
            X_val,
            y_val,
            X_test,
            y_test,
            loss,
        )

        end = datetime.datetime.now()

        return (
            oilPred,
            oilForecast,
            end - start,
            reg_f,
        )

    # DICTIONARY
    return {"LSTM": get_gridsearch_with_LSTM,
        "BiLSTM": get_gridsearch_with_LSTMbidirectional,
        "CNN": get_gridsearch_with_CNN1D,
        "TCN": get_gridsearch_with_TCN, }
