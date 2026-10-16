"""
Code by Bahram Lavi and João Bertini
Description: This code implements several loss functions for training deep learning models.
"""

from tensorflow.keras import backend as K
import tensorflow as tf


def quantile_loss(tau):
    if not (0 < tau < 1):
        raise ValueError("tau must be between 0 and 1.")

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        error = y_true - y_pred
        loss_term_1 = tau * K.maximum(error, 0.0)
        loss_term_2 = (1 - tau) * K.maximum(-error, 0.0)

        return K.mean(loss_term_1 + loss_term_2)

    return loss


def wse_loss(w):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # Residual: positive = underprediction
        #           negative = overprediction
        error = y_true - y_pred

        error_positive = tf.maximum(error, 0.0)
        error_negative = tf.maximum(-error, 0.0)

        squared_error = (
                w * tf.square(error_positive)
                + (1.0 - w) * tf.square(error_negative)
        )

        return tf.reduce_mean(squared_error)

    return loss



def huber_like_asymmetric_loss(w, delta=0.3):
    if not (0 < w < 1):
        raise ValueError("w must be between 0 and 1.")
    if delta <= 0:
        raise ValueError("delta must be positive.")

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        error = y_true - y_pred
        abs_error = K.abs(error)

        positive_error_mask = tf.cast(error > 0, dtype=tf.float32)
        negative_error_mask = tf.cast(error <= 0, dtype=tf.float32)

        squared_loss_part = 0.5 * K.square(error)
        linear_loss_part = delta * (abs_error - 0.5 * delta)

        is_small_error = tf.cast(abs_error < delta, dtype=tf.float32)
        is_large_error = tf.cast(abs_error >= delta, dtype=tf.float32)

        loss_if_positive_and_small = w * squared_loss_part * is_small_error * positive_error_mask
        loss_if_positive_and_large = w * linear_loss_part * is_large_error * positive_error_mask
        loss_if_negative_and_small = (1 - w) * squared_loss_part * is_small_error * negative_error_mask
        loss_if_negative_and_large = (1 - w) * linear_loss_part * is_large_error * negative_error_mask

        total_loss = (loss_if_positive_and_small + loss_if_positive_and_large +
                      loss_if_negative_and_small + loss_if_negative_and_large)

        return K.mean(total_loss)

    return loss


def mse_loss(y_true, y_pred):
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    return K.mean(K.square(y_true - y_pred))


def poisson_loss(y_true, y_pred):
    epsilon = 1e-7
    y_pred = tf.maximum(y_pred, epsilon)
    return tf.reduce_mean(y_pred - y_true * tf.math.log(y_pred))


def calculate_tweedie_loss(p):
    def loss(y_true, y_pred):
        epsilon = 1e-6
        term1 = -y_true * tf.pow(y_pred, 1 - p) / (1 - p)
        term2 = tf.pow(y_pred + epsilon, 2 - p) / (2 - p)
        return tf.reduce_mean(term1 + term2)

    return loss


def fair_loss(c=0.1):
    def loss(y_true, y_pred):
        abs_residual = tf.abs(y_true - y_pred)
        return tf.reduce_mean(c ** 2 * (abs_residual / c - tf.math.log(1.0 + abs_residual / c)))

    return loss


def cauchy_loss(gamma=1.0):
    def loss(y_true, y_pred):
        residual = y_true - y_pred
        return tf.reduce_mean(tf.math.log(1.0 + tf.square(residual / gamma)))

    return loss


def mspe_loss(y_true, y_pred):
    epsilon = 1e-6
    percentage_error = (y_true - y_pred) / (y_true + epsilon)
    return tf.reduce_mean(tf.square(percentage_error))

def smspe_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    numerator = 2.0 * (y_true - y_pred)
    denominator = tf.abs(y_true) + tf.abs(y_pred)

    percentage_error = tf.math.divide_no_nan(
        numerator,
        denominator
    )

    return tf.reduce_mean(
        tf.square(percentage_error)
    )


def rmsle_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    epsilon = 1e-6
    log_diff = tf.math.log1p(y_pred) - tf.math.log1p(y_true)
    return tf.math.sqrt(tf.reduce_mean(tf.math.square(log_diff + epsilon)))


def mape_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float64)
    y_pred = tf.cast(y_pred, tf.float64)
    epsilon = 1e-6
    return tf.reduce_mean(tf.abs((y_true - y_pred) / (y_true + epsilon)))


def smape_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    numerator = 2.0 * tf.abs(y_true - y_pred)
    denominator = tf.abs(y_true) + tf.abs(y_pred)

    smape = tf.math.divide_no_nan(
        numerator,
        denominator
    )

    return tf.reduce_mean(smape)


def log_cosh_loss(y_true, y_pred):
    return tf.reduce_mean(tf.math.log(tf.math.cosh(y_pred - y_true)))


def mae_loss(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))


def tukey_biweight_loss(c):
    if c <= 0:
        raise ValueError("c must be positive.")

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        error = y_true - y_pred
        abs_error = tf.abs(error)
        r = error / c

        inside = (c ** 2 / 6.0) * (1.0 - tf.pow(1.0 - tf.square(r), 3))
        outside = tf.fill(tf.shape(error), tf.cast(c ** 2 / 6.0, tf.float32))
        loss_values = tf.where(abs_error <= c, inside, outside)

        return tf.reduce_mean(loss_values)

    return loss

def call_Losses():
    param_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    param2_values = [0.1, 0.3, 0.7, 0.9]
    param1_val = [0.1, 0.3, 0.5]
    hsl_param1 = [0.25, 0.5, 0.75]
    hsl_param2 = [0.1, 0.3]
    param_tuk = [0.5, 0.7, 0.9]
    all_loss = []

    for val in param_values:
        all_loss.extend([
            {'short_name': f'QL({val})', 'name': f'Quantile Loss (tau={val})', 'function': quantile_loss(tau=val)},
        ])

    for val in param2_values:
        all_loss.extend([
            {'short_name': f'WSE({val})', 'name': f'Weighted Squared Error (w={val})',
             'function': wse_loss(w=val)},
        ])

    for val in param_tuk:
        all_loss.extend([
            {'short_name': f'Tukey({val})', 'name': f'Tukey bitweight (alpha={val})',
             'function': tukey_biweight_loss(c=val)},
        ])


    for val in hsl_param1:
        for val1 in hsl_param2:
            all_loss.extend([
                {'short_name': f'HSL({val}_{val1})', 'name': f'Huber-like Asymmetric Loss (w={val}, delta={val1})',
                 'function': huber_like_asymmetric_loss(w=val,delta=val1)},
            ])


    for val in param1_val:
        all_loss.extend([
            {'short_name': f'Fair({val})', 'name': f'Fair({val})', 'function': fair_loss(c=val)},
            {'short_name': f'Cauchy({val})', 'name': f'Cauchy({val})', 'function': cauchy_loss(gamma=val)},
        ])

    all_loss.extend([
       {'short_name': 'MSE', 'name': 'MSE', 'function': mse_loss},
        {'short_name': 'sMAPE', 'name': 'sMAPE', 'function': smape_loss},
        {'short_name': 'MAE', 'name': 'MAE', 'function': mae_loss},
        {'short_name': 'Log-Cosh', 'name': 'Log-Cosh', 'function': log_cosh_loss},
        {'short_name': 'sMSPE', 'name': 'sMSPE', 'function': smspe_loss},
    ])

    return all_loss