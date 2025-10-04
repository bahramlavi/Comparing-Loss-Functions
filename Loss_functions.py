from tensorflow.keras import backend as K
import tensorflow as tf
tf.keras.backend.set_floatx('float64')

def Call_loss_functions():
    
    def calculate_mse(y_true, y_pred):
      weighted_squared_errors = tf.square(y_pred - y_true)
      mse = tf.reduce_mean(weighted_squared_errors)
      return mse

    def weighted_mse(y_true, y_pred):
        weight = tf.constant(1.5, dtype=tf.float64)
        return weight * tf.square(y_pred - y_true)
        
    def asymmetric_linear_loss(alpha):

        if not (0 < alpha < 1):
            raise ValueError("alpha must be between 0 and 1.")

        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, dtype=tf.float64)
            y_pred = tf.cast(y_pred, dtype=tf.float64)

            error = y_pred - y_true
            positive_error_mask = tf.cast(error > 0, dtype=tf.float64)
            negative_error_mask = tf.cast(error <= 0, dtype=tf.float64)

            weighted_positive_error = alpha * error * positive_error_mask
            weighted_negative_error = (1 - alpha) * -error * negative_error_mask # A - F is -error

            return K.mean(weighted_positive_error + weighted_negative_error)
        return loss

    # 2. Quantile Loss (Pinball Loss)
    def quantile_loss(tau):

        if not (0 < tau < 1):
            raise ValueError("tau must be between 0 and 1.")

        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, dtype=tf.float64)
            y_pred = tf.cast(y_pred, dtype=tf.float64)

            error = y_true - y_pred # A - F
            
            loss_term_1 = tau * K.maximum(error, 0.0)
            loss_term_2 = (1 - tau) * K.maximum(-error, 0.0) # -(A - F) = F - A

            return K.mean(loss_term_1 + loss_term_2)
        return loss

    # 3. Weighted Squared Error (Asymmetric MSE)
    def weighted_squared_error(w):

        if not (0 < w < 1):
            raise ValueError("w must be between 0 and 1.")

        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, dtype=tf.float64)
            y_pred = tf.cast(y_pred, dtype=tf.float64)

            error = y_pred - y_true # F - A
            
            positive_error_mask = tf.cast(error > 0, dtype=tf.float64)
            negative_error_mask = tf.cast(error <= 0, dtype=tf.float64)

            weighted_positive_squared_error = w * K.square(error) * positive_error_mask
            weighted_negative_squared_error = (1 - w) * K.square(error) * negative_error_mask

            return K.mean(weighted_positive_squared_error + weighted_negative_squared_error)
        return loss

    # 4. Huber-like Asymmetric Loss
    def huber_like_asymmetric_loss(w, delta,alpha):

        if not (0 < w < 1):
            raise ValueError("w must be between 0 and 1.")
        if delta <= 0:
            raise ValueError("delta must be positive.")

        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, dtype=tf.float64)
            y_pred = tf.cast(y_pred, dtype=tf.float64)

            error = y_pred - y_true # F - A
            abs_error = K.abs(error)

            positive_error_mask = tf.cast(error > 0, dtype=tf.float64)
            negative_error_mask = tf.cast(error <= 0, dtype=tf.float64)

            squared_loss_part = 0.5 * K.square(error)

            linear_loss_part = delta * (abs_error - 0.5 * delta)


            weighted_positive_error = alpha * error * positive_error_mask
            weighted_negative_error = (1 - alpha) * -error * negative_error_mask 
            
            is_small_error = tf.cast(abs_error < delta, dtype=tf.float64)
            is_large_error = tf.cast(abs_error >= delta, dtype=tf.float64)

            loss_if_positive_and_small = w * squared_loss_part * is_small_error * positive_error_mask
            loss_if_positive_and_large = w * linear_loss_part * is_large_error * positive_error_mask

            loss_if_negative_and_small = (1 - w) * squared_loss_part * is_small_error * negative_error_mask
            loss_if_negative_and_large = (1 - w) * linear_loss_part * is_large_error * negative_error_mask

            total_loss = loss_if_positive_and_small + loss_if_positive_and_large + \
                         loss_if_negative_and_small + loss_if_negative_and_large

            return K.mean(total_loss)
        return loss

    def calculate_wmse(weight):
        
        def loss(y_true, y_pred):
            y_true = tf.cast(y_true, dtype=tf.float64)
            y_pred = tf.cast(y_pred, dtype=tf.float64)
            weighted_squared_errors = weight * tf.square(y_pred - y_true)
            return tf.reduce_mean(weighted_squared_errors)
        return loss

    def mse_loss(y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float64)
        y_pred = tf.cast(y_pred, dtype=tf.float64)
        
        squared_error = K.square(y_true - y_pred)
        
        return K.mean(squared_error)

        
        
    loss_configurations = [
        {'short_name':'WMSE03','name': 'Weighted MSE Loss (weight=0.3)', 'function': calculate_wmse(weight=0.3)},
        {'short_name':'WMSE05','name': 'Weighted MSE Loss (weight=0.5)', 'function': calculate_wmse(weight=0.5)},
        {'short_name':'WMSE07','name': 'Weighted MSE Loss (weight=0.7)', 'function': calculate_wmse(weight=0.7)},
        {'short_name':'WMSE09','name': 'Weighted MSE Loss (weight=0.9)', 'function': calculate_wmse(weight=0.9)},
        {'short_name':'WMSE01','name': 'Weighted MSE Loss (weight=0.1)', 'function': calculate_wmse(weight=0.1)},
        {'short_name':'AsL09','name': 'Asymmetric Linear Loss (alpha=0.9)', 'function': asymmetric_linear_loss(alpha=0.9)},
        {'short_name':'AsL07','name': 'Asymmetric Linear Loss (alpha=0.7)', 'function': asymmetric_linear_loss(alpha=0.7)},
        {'short_name':'AsL05','name': 'Asymmetric Linear Loss (alpha=0.5)', 'function': asymmetric_linear_loss(alpha=0.5)},
        {'short_name':'AsL03','name': 'Asymmetric Linear Loss (alpha=0.3)', 'function': asymmetric_linear_loss(alpha=0.3)},
        {'short_name':'AsL01','name': 'Asymmetric Linear Loss (alpha=0.1)', 'function': asymmetric_linear_loss(alpha=0.1)},
        {'short_name':'QL01','name': 'Quantile Loss (tau=0.1)', 'function': quantile_loss(tau=0.1)},
        {'short_name':'QL03','name': 'Quantile Loss (tau=0.3)', 'function': quantile_loss(tau=0.3)},
        {'short_name':'QL05','name': 'Quantile Loss (tau=0.5)', 'function': quantile_loss(tau=0.5)},
        {'short_name':'QL07','name': 'Quantile Loss (tau=0.7)', 'function': quantile_loss(tau=0.7)},
        {'short_name':'QL09','name': 'Quantile Loss (tau=0.9)', 'function': quantile_loss(tau=0.9)},
        {'short_name':'WSE09','name': 'Weighted Squared Error (w=0.9)', 'function': weighted_squared_error(w=0.9)},
        {'short_name':'WSE07','name': 'Weighted Squared Error (w=0.7)', 'function': weighted_squared_error(w=0.7)},
        {'short_name':'WSE05','name': 'Weighted Squared Error (w=0.5)', 'function': weighted_squared_error(w=0.5)},
        {'short_name':'WSE03','name': 'Weighted Squared Error (w=0.3)', 'function': weighted_squared_error(w=0.3)},
        {'short_name':'WSE01','name': 'Weighted Squared Error (w=0.1)', 'function': weighted_squared_error(w=0.1)},
        {'short_name':'HSL01','name': 'Huber-like Asymmetric Loss (w=0.1, delta=1.0)', 'function': huber_like_asymmetric_loss(w=0.1,delta=0.1, alpha=1.0)},
        {'short_name':'HSL03','name': 'Huber-like Asymmetric Loss (w=0.3, delta=1.0)', 'function': huber_like_asymmetric_loss(w=0.3,delta=0.3, alpha=1.0)},
        {'short_name':'HSL05','name': 'Huber-like Asymmetric Loss (w=0.5, delta=1.0)', 'function': huber_like_asymmetric_loss(w=0.5,delta=0.5, alpha=1.0)},
        {'short_name':'HSL07','name': 'Huber-like Asymmetric Loss (w=0.7, delta=1.0)', 'function': huber_like_asymmetric_loss(w=0.7,delta=0.7, alpha=1.0)},
        {'short_name':'HSL09','name': 'Huber-like Asymmetric Loss (w=0.9, delta=1.0)', 'function': huber_like_asymmetric_loss(w=0.9,delta=0.9, alpha=1.0)},
        {'short_name':'MSE','name': 'MSE', 'function': mse_loss}
    ]       

    return loss_configurations