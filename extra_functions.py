# -*- coding: utf-8 -*-
"""
Created on Sun Mar  8 18:03:58 2020

@author: World peace
"""

def configure_optimizer(flags, learning_rate):
    """Configures the optimizer used for training.
    Args:
      learning_rate: A scalar or `Tensor` learning rate.
    Returns:
      An instance of an optimizer.
    """
    if flags.optimizer == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=flags.adadelta_rho,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=flags.adagrad_initial_accumulator_value)
    elif flags.optimizer == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=flags.adam_beta1,
            beta2=flags.adam_beta2,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=flags.ftrl_learning_rate_power,
            initial_accumulator_value=flags.ftrl_initial_accumulator_value,
            l1_regularization_strength=flags.ftrl_l1,
            l2_regularization_strength=flags.ftrl_l2)
    elif flags.optimizer == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=flags.momentum,
            name='Momentum')
    elif flags.optimizer == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=flags.rmsprop_decay,
            momentum=flags.rmsprop_momentum,
            epsilon=flags.opt_epsilon)
    elif flags.optimizer == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', flags.optimizer)
    return optimizer