import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import wandb
import math

from ..core import WORKING_DIR, red, green, blue



def optimizer_factory(kwargs, lr_scheduler):
    optimizer_name = kwargs._target_
    if optimizer_name == 'AdamW':
        optimizer =  tfa.optimizers.AdamW(
            beta_1=kwargs.beta_1,
            beta_2=kwargs.beta_2,
            epsilon=kwargs.epsilon,
            weight_decay=kwargs.weight_decay,
            clipnorm=kwargs.max_grad_norm,
            amsgrad=False, # Does not work on TPU
            learning_rate=lr_scheduler,
        )
    if kwargs.use_swa:
        print(red('Using SWA'))
        optimizer = tfa.optimizers.SWA(optimizer)
    if kwargs.use_lookahead:
        print(red('Using Lookahead'))
        optimizer = tfa.optimizers.Lookahead(optimizer)
    if 'average_decay' in kwargs: 
        print(blue('Using moving average'))
        optimizer = tfa.optimizers.MovingAverage(
            optimizer, 
            average_decay=kwargs.average_decay, 
            dynamic_decay=kwargs.dynamic_decay
        )

    return optimizer


def callbacks_factory(kwargs):
    monitor, mode = kwargs.monitor_mode
    common_kwargs = {
        'monitor': monitor,
        'mode': mode,
        'verbose': True,
    }
    callbacks = []
    if 'checkpoint_file' in kwargs:
        print(f'tf.keras.callbacks.ModelCheckpoint: Saving model checkpoints at {kwargs.checkpoint_file}')
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=str(WORKING_DIR/kwargs.checkpoint_file),
            save_weights_only=True,
            save_best_only=True,
            **common_kwargs,
        )
        callbacks.append(model_checkpoint_callback)
    if 'early_stop' in kwargs:
        print(f'tf.keras.callbacks.EarlyStopping: Will stop training if metric {kwargs.monitor_mode[0]} does not improve after {kwargs.early_stop} epochs')
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=kwargs.early_stop,
            restore_best_weights=True,
            **common_kwargs,
        )
        callbacks.append(early_stopping_callback)
    if 'max_train_hours' in kwargs:
        print(f'tfa.callbacks.TimeStopping: Will stop training after {kwargs.max_train_hours} hours')
        time_stopping_callback = tfa.callbacks.TimeStopping(
            seconds=int(kwargs.max_train_hours*3600), verbose=common_kwargs['verbose'],
        )
        callbacks.append(time_stopping_callback)
    if 'reduce_lr' in kwargs:
        print(f'tf.keras.callbacks.ReduceLROnPlateau: Warning: only works in float LR')
        reduce_lr_callback =  tf.keras.callbacks.ReduceLROnPlateau(
            factor=kwargs.reduce_lr['factor'],
            patience=kwargs.reduce_lr['patience'],
            min_delta=0,
            min_lr=1e-8,
            **common_kwargs,
        )
        callbacks.append(reduce_lr_callback)
    if 'terminate_on_nan' in kwargs:
        callbacks.append(tf.keras.callbacks.TerminateOnNaN())
    if 'tqdm_bar' in kwargs:
        callbacks.append(tfa.callbacks.TQDMProgressBar())

    return callbacks

def get_wandb_callback(callbacks_kwargs, train_ds, valid_ds, valid_steps):
    try:
        monitor, mode = callbacks_kwargs.monitor_mode
        wandb_callback = wandb.keras.WandbCallback(
            monitor=monitor, mode=mode,
            save_model=True, save_graph=True,
            save_weights_only=True,
            log_weights=True,
            log_gradients=True,
            training_data=train_ds,
            validation_data=valid_ds,
            validation_steps=valid_steps,
        )
        return [wandb_callback]
    except Exception as e:
        try:
            print('wandb exception:', e)
            print('using lightweight version of wandb callback')
            wandb_callback = wandb.keras.WandbCallback(
                monitor=monitor, mode=mode,
                save_model=False, save_graph=True,
            )
            return [wandb_callback]
        except Exception as e:
            print('wandb_callback: ', e)
            return []

class WarmUp(tf.keras.optimizers.schedules.LearningRateSchedule):
    'Applies warmup schedule on the given lr schedule function'
    def __init__(self, warmup_lr, lr_scheduler, warmup_steps, power=1.0):
        super().__init__()
        self.warmup_lr = warmup_lr
        self.lr_scheduler = lr_scheduler
        self.warmup_steps = warmup_steps
        self.power = power

    def __call__(self, step):
        with tf.name_scope('WarmUp') as name:
            global_step_float = tf.cast(step, tf.float32)
            warmup_steps_float = tf.cast(self.warmup_steps, tf.float32)
            warmup_percent_done = global_step_float / warmup_steps_float
            warmup_learning_rate = self.warmup_lr * tf.math.pow(warmup_percent_done, self.power)
            return tf.cond(
                global_step_float < warmup_steps_float,
                lambda: warmup_learning_rate,
                lambda: self.lr_scheduler(step - self.warmup_steps),
                name=name,
            )

    def get_config(self):
        return {
            'warmup_lr': self.warmup_lr,
            'lr_scheduler': self.lr_scheduler,
            'warmup_steps': self.warmup_steps,
            'power': self.power,
        }

class CosineDecayRestarts(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Cosine decay with restarts
    gamma, step_multiply, min_lr_ratio=1e-2,
    """

    def __init__(self, lr, first_decay_steps, step_gamma=2, gamma=1, min_lr_ratio=0):
        super().__init__()
        self.initial_learning_rate = lr
        self.first_decay_steps = first_decay_steps
        self._t_mul = step_gamma
        self._m_mul = gamma
        self.alpha = min_lr_ratio

    def __call__(self, step):
        with tf.name_scope('SGDRDecay') as name:
            initial_learning_rate = tf.convert_to_tensor(
                self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            first_decay_steps = tf.cast(self.first_decay_steps, dtype)
            alpha = tf.cast(self.alpha, dtype)
            t_mul = tf.cast(self._t_mul, dtype)
            m_mul = tf.cast(self._m_mul, dtype)

            global_step_recomp = tf.cast(step, dtype)
            completed_fraction = global_step_recomp / first_decay_steps

        def compute_step(completed_fraction, geometric=False):
            """Helper for `cond` operation."""
            if geometric:
                i_restart = tf.floor(
                    tf.math.log(1.0 - completed_fraction * (1.0 - t_mul)) /
                    tf.math.log(t_mul))

                sum_r = (1.0 - t_mul**i_restart) / (1.0 - t_mul)
                completed_fraction = (completed_fraction - sum_r) / t_mul**i_restart

            else:
                i_restart = tf.floor(completed_fraction)
                completed_fraction -= i_restart
            return i_restart, completed_fraction

        i_restart, completed_fraction = tf.cond(
            tf.equal(t_mul, 1.0),
            lambda: compute_step(completed_fraction, geometric=False),
            lambda: compute_step(completed_fraction, geometric=True))

        m_fac = m_mul**i_restart
        cosine_decayed = 0.5 * m_fac * (1.0 + tf.cos(
            tf.constant(math.pi) * completed_fraction))
        decayed = (1 - alpha) * cosine_decayed + alpha
        return tf.multiply(initial_learning_rate, decayed, name=name)

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "first_decay_steps": self.first_decay_steps,
            "t_mul": self._t_mul,
            "m_mul": self._m_mul,
            "alpha": self.alpha,
            "name": 'CosineLRDecay', 
        }

def plot_first_epoch(lr_scheduler, train_steps, checkpoints_per_epoch):
    plt.rcParams['figure.figsize'] = (20,3) # TODO: Move globally
    steps = list(range(0, train_steps, train_steps//checkpoints_per_epoch+1))
    _ = plt.plot([lr_scheduler(x) for x in range(train_steps)], markevery=steps, marker='o')


def lr_scheduler_factory(warmup_epochs, warmup_power, lr_cosine, train_steps): 
    non_warmup_steps = int(train_steps * (1-warmup_epochs))
    warmup_steps = int(train_steps * warmup_epochs)
    first_decay_steps = non_warmup_steps//sum(lr_cosine.step_gamma**i for i in range(1, lr_cosine.num_cycles))+1
    if warmup_epochs >= 1: 
        first_decay_steps = train_steps 

    min_lr_ratio = lr_cosine.min_lr / lr_cosine.max_lr
    lr_scheduler = CosineDecayRestarts(
        lr_cosine.max_lr,
        first_decay_steps,
        lr_cosine.step_gamma,
        lr_cosine.lr_gamma,
        min_lr_ratio,
    )

    # Add warmup to scheduler
    lr_scheduler = WarmUp(
        warmup_lr=lr_cosine.max_lr,
        lr_scheduler=lr_scheduler,
        warmup_steps=warmup_steps,
        power=warmup_power,
    )
    return lr_scheduler


def lr_scheduler_factory_v1(kwargs):
    non_warmup_steps = kwargs.train_steps * (1-kwargs.warmup.ratio)
    warmup_steps = kwargs.warmup.ratio * kwargs.train_steps
    if kwargs._target_ == 'constant':
        print('Using constant lr')
        lr_scheduler = lambda step: kwargs.lr
    elif kwargs._target_ == 'ExponentialCyclicalLearningRate':
        print('Using exponential cyclic LR')
        step_size = int(kwargs.train_steps/(2*kwargs.num_cycles))
        lr_scheduler = tfa.optimizers.ExponentialCyclicalLearningRate(
            initial_learning_rate=kwargs.min_lr,
            maximal_learning_rate=kwargs.max_lr,
            gamma=kwargs.gamma,
            step_size=step_size,
            scale_mode='cycle',
        )
    elif kwargs._target_ == 'CosineDecayRestarts':
        first_decay_steps = non_warmup_steps//sum(kwargs.step_gamma**i for i in range(1, kwargs.num_cycles))
        lr_scheduler = CosineDecayRestarts(kwargs.lr, first_decay_steps, kwargs.step_gamma, kwargs.lr_gamma, kwargs.min_lr_ratio)

    # Add Warmup to LR Scheduler
    if 'warmup' in kwargs:
        print('Adding warmup to lr scheduler')
        lr_scheduler = WarmUp(
            warmup_lr=kwargs.warmup.lr,
            lr_scheduler=lr_scheduler,
            warmup_steps=warmup_steps,
            power=kwargs.warmup.power,
        )

    return lr_scheduler




