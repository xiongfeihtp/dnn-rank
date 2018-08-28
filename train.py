'''
@author: xiongfei
@license: (C) Copyright 2013-2017, Node Supply Chain Manager Corporation Limited.
@contact: 386344277@qq.com
@file: train.py
@time: 2018/8/27 下午3:21
@desc: shanghaijiaotong university
'''
import tensorflow as tf
import os
import shutil
import numpy as np
import time
from collections import OrderedDict
import yaml

ModeKeys = tf.estimator.ModeKeys
tf.logging.set_verbosity(tf.logging.INFO)


def elapse_time(start_time):
    return round((time.time() - start_time) / 60)


# os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # 这是默认的显示等级，显示所有信息
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'  # 只显示 warning 和 Error


# os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

def parse_fn(field_delim='\t', na_value=''):
    feature_conf = load_conf('feature.yaml')
    feature_list = list(load_conf('schema.yaml').values())
    feature = feature_list
    feature_unused = []

    def column_to_csv_defaults():
        """parse columns to record_defaults param in tf.decode_csv func
        Return:
            OrderedDict {'feature name': [''],...}
        """
        csv_defaults = OrderedDict()
        csv_defaults['is_click'] = [0]
        for f in feature:
            if f in feature_conf:  # used features
                conf = feature_conf[f]
                if conf['type'] == 'category':
                    if conf['transform'] == 'identity':  # identity category column need int type
                        csv_defaults[f] = [0]
                    else:
                        csv_defaults[f] = [0]
                else:
                    csv_defaults[f] = [0.0]  # 0.0 for float32
            else:  # unused feature
                feature_unused.append(f)
                csv_defaults[f] = ['']
        return csv_defaults

    csv_defaults = column_to_csv_defaults()

    def parser(value):
        """Parse train and eval data with label
        Args:
            value: Tensor("arg0:0", shape=(), dtype=string)
        """
        # `tf.decode_csv` return rank 0 Tensor list: <tf.Tensor 'DecodeCSV:60' shape=() dtype=string>
        # na_value fill with record_defaults
        columns = tf.decode_csv(
            value, record_defaults=list(csv_defaults.values()),
            field_delim=field_delim, use_quote_delim=False, na_value=na_value)
        features = dict(zip(csv_defaults.keys(), columns))
        temp = features.copy()
        for f, tensor in temp.items():
            if f in feature_unused:
                features.pop(f)  # remove unused features
        # csv parse as string
        labels = tf.equal(temp.pop('is_click'), 1)
        return features, labels

    return parser


def input_fn(data_file, mode, batch_size):
    input_conf = load_conf('train.yaml')['train']
    num_parallel_calls = input_conf['num_parallel_calls']
    shuffle_buffer_size = input_conf['shuffle_buffer_size']
    tf.logging.info("Parsing input file: {}".format(data_file))
    dataset = tf.data.TextLineDataset(data_file)
    dataset = dataset.map(parse_fn(),
                          num_parallel_calls=num_parallel_calls)
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, seed=123)

    dataset = dataset.prefetch(2 * batch_size)
    dataset = dataset.batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()


def train_and_eval(model):
    train_conf = load_conf('train.yaml')['train']
    train_epochs = train_conf['train_epochs']
    train_data = train_conf['train_data']
    eval_data = train_conf['eval_data']
    test_data = train_conf['test_data']
    batch_size = train_conf['batch_size']
    epochs_per_eval = train_conf['epochs_per_eval']

    for n in range(train_epochs):
        tf.logging.info('=' * 30 + ' START EPOCH {} '.format(n + 1) + '=' * 30 + '\n')
        t0 = time.time()
        tf.logging.info('<EPOCH {}>: Start training {}'.format(n + 1, train_data))
        model.train(input_fn=lambda: input_fn(train_data, ModeKeys.TRAIN, batch_size),
                    hooks=None,
                    steps=None,
                    max_steps=None,
                    saving_listeners=None)
        tf.logging.info('<EPOCH {}>: Finish training {}, take {} mins'.format(n + 1, train_data, elapse_time(t0)))
        print('-' * 80)

        tf.logging.info('<EPOCH {}>: Start evaluating {}'.format(n + 1, eval_data))
        t0 = time.time()
        results = model.evaluate(
            input_fn=lambda: input_fn(eval_data, ModeKeys.EVAL, batch_size),
            steps=None,  # Number of steps for which to evaluate model.
            hooks=None,
            checkpoint_path=None,  # latest checkpoint in model_dir is used.
            name=None)

        tf.logging.info(
            '<EPOCH {}>: Finish evaluation {}, take {} mins'.format(n + 1, eval_data, elapse_time(t0)))
        print('-' * 80)
        # Display evaluation metrics
        for key in sorted(results):
            print('{}: {}'.format(key, results[key]))


from tensorflow.contrib import layers


# 定义训练方法，包括优化方法和梯度预处理
def get_train_op_fn(loss, params):
    return layers.optimize_loss(
        loss=loss,
        global_step=tf.train.get_global_step(),
        optimizer=tf.train.AdamOptimizer(),
        learning_rate=params['learning_rate']
    )


# 定义评价指标
def get_eval_metric_ops(labels, predictions):
    """Return a dict of the evaluation Ops.
    Args:
        labels (Tensor): Labels tensor for training and evaluation.
        predictions (Tensor): Predictions Tensor.
    Returns:
        Dict of metric results keyed by name.
    """
    return {
        'auc': tf.metrics.auc(
            labels=labels,
            predictions=predictions,
            name='auc')
    }


def model(wide_features, deep_features, dense_features, is_training):
    train_conf = load_conf('train.yaml')['train']
    model_type = train_conf['model_type']
    model_conf = load_conf('model.yaml')
    activation = model_conf['activation']
    initializer = model_conf['initializer']
    if initializer == 'xavier':
        initializer_fn = tf.contrib.layers.xavier_initializer()
    else:
        raise TypeError('initializer_fn {} is not implement'.format(initializer))

    if activation == 'relu':
        activation_fn = tf.nn.relu
    elif activation == 'selu':
        activation_fn = tf.nn.selu
    else:
        raise TypeError('activation_fn {} is not implement'.format(activation))

    if model_type == 'wdl':
        dnn_hidden_units = model_conf['dnn_hidden_units']
        deep_input = tf.concat([deep_features, dense_features], axis=1)

        for dim in dnn_hidden_units:
            deep_input = tf.layers.dense(deep_input, dim, activation=activation_fn,
                                         kernel_initializer=initializer_fn)

        deep_input = tf.layers.dense(deep_input, 1, activation=None,
                                     kernel_initializer=initializer_fn)

        return deep_input + tf.reduce_sum(wide_features, axis=1, keep_dims=True)

    elif model_type == 'xcross':
        deep_input = tf.split(deep_features, num_or_size_splits=model_conf['field_num'], axis=1)
        deep_input = tf.concat([tf.expand_dims(feature, axis=1) for feature in deep_input], axis=1)
        cross_layer_size = model_conf['cross_layer_size']
        final_result = []
        final_len = 0
        embed_dim = model_conf['embed_dim']
        field_nums = [model_conf['field_num']]
        cin_layers = [deep_input]
        for idx, layer_size in enumerate(cross_layer_size):
            dot_result = tf.einsum('ipk,iqk->ipqk', cin_layers[0], cin_layers[-1])
            dot_result = tf.reshape(dot_result, shape=[-1, embed_dim, field_nums[0] * field_nums[-1]])
            filters = tf.get_variable(initializer=initializer_fn, name='f_' + str(idx),
                                      shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                      dtype=tf.float32)

            # [batch_size, embed_dim, layers_size]
            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
            final_len += layer_size
            field_nums.append(layer_size)
            final_result.append(curr_out)
            curr_out = tf.transpose(curr_out, [0, 2, 1])
            cin_layers.append(curr_out)
        result = tf.concat(final_result, axis=2)
        result = tf.reduce_sum(result, axis=1)  # [batch_size, layer_size]

        input = tf.concat([wide_features, result, dense_features], axis=1)
        dnn_hidden_units = model_conf['dnn_hidden_units']
        for dim in dnn_hidden_units:
            input = tf.layers.dense(input, dim, activation=activation_fn,
                                    kernel_initializer=initializer_fn)
        input = tf.layers.dense(input, 1, activation=None, kernel_initializer=initializer_fn)
        return input

    elif model_type == 'dcn':
        # cross
        # [batch_size, dim, 1]
        with tf.variable_scope('dcn'):
            cross_dcn_layer_size = model_conf['cross_dcn_layer_size']
            x_0 = tf.expand_dims(deep_features, axis=2)
            cross = x_0
            for i in range(cross_dcn_layer_size):
                cross = tf.layers.dense(tf.matmul(x_0, cross, transpose_b=True), 1, activation=None,
                                        kernel_initializer=initializer_fn, name='dcn_cross_{}'.format(i)) + x_0
            cross = tf.squeeze(cross, axis=2)

        with tf.variable_scope('dnn'):
            dnn_hidden_units = model_conf['dnn_hidden_units']
            deep_input = tf.concat([deep_features, dense_features], axis=1)

            for dim in dnn_hidden_units:
                deep_input = tf.layers.dense(deep_input, dim, activation=activation_fn,
                                             kernel_initializer=initializer_fn)

        deep_input = tf.layers.dense(tf.concat([deep_input, cross], axis=1), 1, activation=None,
                                     kernel_initializer=initializer_fn)
        return deep_input

    elif model_type == 'deepfm':
        with tf.variable_scope('fm'):
            # [batch_size, 1]
            y_first_order = tf.reduce_sum(wide_features, axis=1, keep_dims=True)
            second_input = tf.split(deep_features, num_or_size_splits=model_conf['field_num'], axis=1)
            # [batch_size, field_num, dim]
            second_input = tf.concat([tf.expand_dims(feature, axis=1) for feature in second_input], axis=1)

            # [batch_size, dim]
            sum_square = tf.square(tf.reduce_sum(second_input, axis=1))
            square_sum = tf.reduce_sum(tf.square(second_input), axis=1)
            # [batch_size, 1]
            y_second_order = 0.5 * tf.reduce_sum(tf.subtract(sum_square, square_sum), axis=1, keep_dims=True)

        with tf.variable_scope('dnn'):
            dnn_hidden_units = model_conf['dnn_hidden_units']
            deep_input = tf.concat([deep_features, dense_features], axis=1)

            for dim in dnn_hidden_units:
                deep_input = tf.layers.dense(deep_input, dim, activation=activation_fn,
                                             kernel_initializer=initializer_fn)

            deep_input = tf.layers.dense(deep_input, 1, activation=None,
                                         kernel_initializer=initializer_fn)
        return deep_input + y_first_order + y_second_order

    elif model_type == 'xdeepfm':
        with tf.variable_scope('xdeepfm'):
            deep_input = tf.split(deep_features, num_or_size_splits=model_conf['field_num'], axis=1)
            deep_input = tf.concat([tf.expand_dims(feature, axis=1) for feature in deep_input], axis=1)
            cross_layer_size = model_conf['cross_layer_size']
            final_result = []
            final_len = 0
            embed_dim = model_conf['embed_dim']
            field_nums = [model_conf['field_num']]
            cin_layers = [deep_input]
            for idx, layer_size in enumerate(cross_layer_size):
                dot_result = tf.einsum('ipk,iqk->ipqk', cin_layers[0], cin_layers[-1])
                dot_result = tf.reshape(dot_result, shape=[-1, embed_dim, field_nums[0] * field_nums[-1]])
                filters = tf.get_variable(initializer=initializer_fn, name='f_' + str(idx),
                                          shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                          dtype=tf.float32)

                # [batch_size, embed_dim, layers_size]
                curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')
                final_len += layer_size
                field_nums.append(layer_size)
                final_result.append(curr_out)
                curr_out = tf.transpose(curr_out, [0, 2, 1])
                cin_layers.append(curr_out)
            result = tf.concat(final_result, axis=2)
            result = tf.reduce_sum(result, axis=1)  # [batch_size, layer_size]
            y_xdeepfm = tf.layers.dense(result, 1, activation=None,
                                        kernel_initializer=initializer_fn)
        with tf.variable_scope('dnn'):
            dnn_hidden_units = model_conf['dnn_hidden_units']
            deep_input = tf.concat([deep_features, dense_features], axis=1)
            for dim in dnn_hidden_units:
                deep_input = tf.layers.dense(deep_input, dim, activation=activation_fn,
                                             kernel_initializer=initializer_fn)
            y_dnn = tf.layers.dense(deep_input, 1, activation=None,
                                    kernel_initializer=initializer_fn)

        return y_xdeepfm + y_dnn + tf.reduce_sum(wide_features, axis=1, keep_dims=True)

    else:
        raise TypeError('model_type {} not implement'.format(model_type))


def model_fn(features, labels, mode, params, config):
    """Model function used in the estimator.
    Args:
        features (Tensor): Input features to the model.
        labels (Tensor): Labels tensor for training and evaluation.
        mode (ModeKeys): Specifies if training, evaluation or prediction.
        params (HParams): hyperparameters.
    Returns:
        (EstimatorSpec): Model to be run by Estimator.
    """
    is_training = mode == ModeKeys.TRAIN
    # features tensor
    wide_features = tf.feature_column.input_layer(features, params['wide_columns'])
    features = tf.feature_column.input_layer(features, params['deep_columns'])
    deep_features, dense_features = tf.split(features, [params['deep_dim'], params['dense_dim']], axis=1)
    # Define model's architecture
    print('wide_features: {}'.format(wide_features.get_shape()))
    print('deep_features: {}'.format(deep_features.get_shape()))
    print('dense_features: {}'.format(dense_features.get_shape()))

    logits = model(wide_features, deep_features, dense_features, is_training=is_training)
    predictions = tf.nn.sigmoid(logits)

    # Loss, training and eval operations are not needed during inference.

    loss = None
    train_op = None
    eval_metric_ops = {}
    if mode != ModeKeys.PREDICT:
        labels = tf.expand_dims(labels, axis=1)
        loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.cast(labels, tf.int32), logits=logits)
        train_op = get_train_op_fn(loss, params)
        eval_metric_ops = get_eval_metric_ops(labels, predictions)

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops
    )


# wide columns
categorical_column_with_identity = tf.feature_column.categorical_column_with_identity
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
categorical_column_with_vocabulary_list = tf.feature_column.categorical_column_with_vocabulary_list
crossed_column = tf.feature_column.crossed_column
bucketized_column = tf.feature_column.bucketized_column
# deep columns
embedding_column = tf.feature_column.embedding_column
indicator_column = tf.feature_column.indicator_column
numeric_column = tf.feature_column.numeric_column


def build_model_columns():
    def embedding_dim(dim):
        return int(np.power(2, np.ceil(np.log(dim ** 0.25))))

    def normalizer_fn_builder(scaler, normalization_params):
        """normalizer_fn builder"""
        if scaler == 'min_max':
            return lambda x: (x - normalization_params[0]) / (x - normalization_params[1])
        elif scaler == 'standard':
            return lambda x: (x - normalization_params[0]) / normalization_params[1]
        else:
            return lambda x: tf.log(x)

    feature_conf = load_conf('feature.yaml')
    wide_columns = []
    deep_columns = []

    wide_dim = 0
    deep_dim = 0
    dense_dim = 0

    for feature, conf in feature_conf.items():
        f_type, f_tran, f_param = conf['type'], conf['transform'], conf['parameter']
        if f_type == 'category':
            if f_tran == 'hash_bucket':
                hash_bucket_size = f_param['hash_bucket_size']
                embed_dim = f_param['embed_size']
                col = categorical_column_with_identity(feature, num_buckets=hash_bucket_size, default_value=0)
                wide_columns.append(embedding_column(col,
                                                     dimension=1,
                                                     combiner='sum',
                                                     initializer=None,
                                                     ckpt_to_load_from=None,
                                                     tensor_name_in_ckpt=None,
                                                     max_norm=None,
                                                     trainable=True))

                deep_columns.append(embedding_column(col,
                                                     dimension=embed_dim,
                                                     combiner='sum',
                                                     initializer=None,
                                                     ckpt_to_load_from=None,
                                                     tensor_name_in_ckpt=None,
                                                     max_norm=None,
                                                     trainable=True))

                wide_dim += 1
                deep_dim += embed_dim

        # 连续值
        else:
            normalizaton, boundaries = f_param["normalization"], f_param["boundaries"]
            if f_tran is None:
                normalizer_fn = None
            else:
                normalizer_fn = normalizer_fn_builder(f_tran, tuple(normalizaton))
            col = numeric_column(feature,
                                 shape=(1,),
                                 default_value=0,  # default None will fail if an example does not contain this column.
                                 dtype=tf.float32,
                                 normalizer_fn=normalizer_fn)
            if boundaries:  # whether include continuous features in wide part
                wide_columns.append(bucketized_column(col, boundaries=boundaries))
                wide_dim += (len(boundaries) + 1)

            deep_columns.append(col)
            dense_dim += 1
            # add columns logging info

    tf.logging.info('Build total {} wide columns'.format(len(wide_columns)))
    for col in wide_columns:
        tf.logging.debug('Wide columns: {}'.format(col))
    tf.logging.info('Build total {} deep columns'.format(len(deep_columns)))
    for col in deep_columns:
        tf.logging.debug('Deep columns: {}'.format(col))
    tf.logging.info('Wide input dimension is: {}'.format(wide_dim))
    tf.logging.info('Deep input dimension is: {}'.format(deep_dim))
    tf.logging.info('Dense input dimension is: {}'.format(dense_dim))
    return wide_columns, deep_columns, wide_dim, deep_dim, dense_dim


def build_estimator(model_dir):
    wide_columns, deep_columns, _, deep_dim, dense_dim = build_model_columns()
    config = tf.ConfigProto(device_count={"GPU": 1},  # limit to GPU usage
                            inter_op_parallelism_threads=0,
                            intra_op_parallelism_threads=0,
                            log_device_placement=True)

    run_config = load_conf('train.yaml')['runconfig']
    run_config = tf.estimator.RunConfig(**run_config).replace(session_config=config)
    model_config = load_conf('model.yaml')
    params = {
        'wide_columns': wide_columns,
        'deep_columns': deep_columns,
        'deep_dim': deep_dim,
        'dense_dim': dense_dim,
        'learning_rate': model_config['learning_rate']
    }

    return tf.estimator.Estimator(
        model_dir=model_dir,
        model_fn=model_fn,
        params=params,
        config=run_config
    )


def load_conf(filename):
    with open('./conf/' + filename, 'r') as f:
        return yaml.load(f)


def run():
    train_conf = load_conf('train.yaml')['train']
    print("Using train config:")
    for k, v in train_conf.items():
        print('{}: {}'.format(k, v))

    model_conf = load_conf('model.yaml')
    print("Using model config:")
    for k, v in model_conf.items():
        print('{}: {}'.format(k, v))

    model_dir = os.path.join(train_conf['model_dir'], train_conf['model_type'])

    if not train_conf['keep_train']:
        shutil.rmtree(model_dir, ignore_errors=True)
        print("remove model directory: {}".format(model_dir))

    estimator = build_estimator(model_dir)
    tf.logging.info("Build estimator: {}".format(estimator))
    train_and_eval(estimator)


if __name__ == '__main__':
    run()
