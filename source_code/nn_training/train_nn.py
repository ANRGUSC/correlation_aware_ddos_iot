import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import random
import os
from pickle import dump
import matplotlib.pyplot as plt
import glob
from datetime import datetime
import gc

sys.path.append('../')
sys.path.append('../../')
sys.path.append('./source_code/')

import project_config as CONFIG


def prepare_output_directory(output_path):
    dir_name = str(os.path.dirname(output_path))
    os.system('rm -rf ' + dir_name)
    os.system('mkdir -p ' + dir_name)


def load_dataset(path):
    data = pd.read_csv(path)
    return data


def upsample_dataset(X, y, num_labels):
    df = pd.DataFrame(X)
    df['ATTACKED'] = y
    print(df['ATTACKED'].value_counts())
    attacked_data = df.loc[df['ATTACKED'] == 1]
    not_attacked_data = df.loc[df['ATTACKED'] == 0]
    attacked_data = resample(attacked_data, replace=True, n_samples=not_attacked_data.shape[0], random_state=10)
    df = pd.concat([not_attacked_data, attacked_data])
    print(df['ATTACKED'].value_counts())

    X = np.array(df.iloc[:,0:-num_labels])
    y = np.array(df.iloc[:,-num_labels:])

    return X, y


def calculate_dataset_step(dataset):
    attack_ratios = dataset['ATTACK_RATIO'].unique()
    attack_start_times = dataset['ATTACK_START_TIME'].unique()
    attack_durations = dataset['ATTACK_DURATION'].unique()
    k_list = dataset['ATTACK_PARAMETER'].unique()
    nodes = dataset['NODE'].unique()
    dataset = dataset.sort_values(by=['NODE', 'ATTACK_RATIO', 'ATTACK_START_TIME', 'ATTACK_DURATION',
                                      'ATTACK_PARAMETER', 'TIME']).reset_index(drop=True)
    for node in nodes:
        for k in k_list:
            for attack_ratio in attack_ratios:
                for attack_start_time in attack_start_times:
                    for attack_duration in attack_durations:
                        temp = dataset.loc[(dataset['NODE'] == node) &
                                           (dataset['ATTACK_RATIO'] == attack_ratio) &
                                           (dataset['ATTACK_START_TIME'] == attack_start_time) &
                                           (dataset['ATTACK_DURATION'] == attack_duration) &
                                           (dataset['ATTACK_PARAMETER'] == k)]
                        if temp.shape[0] == 0:
                            continue
                        return temp.shape[0]


def get_dataset_input_output(nn_model, architecture, data_type, dataset, selected_nodes_for_correlation, num_labels,
                             time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled):
    # set up the required features to be used for training/validating
    temp_columns = []
    if use_time_hour is True:
        temp_columns.append('TIME_HOUR')
    if architecture == 'multiple_models_with_correlation' or architecture == 'one_model_with_correlation':
        for node in selected_nodes_for_correlation:
            temp_columns.append('PACKET_' + str(node))
    elif architecture == 'multiple_models_without_correlation' or architecture == 'one_model_without_correlation':
        temp_columns.append('PACKET')
    if architecture == 'one_model_with_correlation' or architecture == 'one_model_without_correlation':
        if use_onehot is True:
            for node in selected_nodes_for_correlation:
                temp_columns.append('NODE_' + str(node))
        else:
            temp_columns.append('NODE')
    temp_columns.append('ATTACKED')

    temp = dataset[temp_columns]
    X = temp.iloc[:,0:-num_labels]
    X = np.asarray(X).astype(float)
    # y = temp.iloc[:,-num_labels:]
    # y = np.asarray(y).astype(float)
    if data_type == 'train':
        scaler = StandardScaler()
        scaler.fit_transform(X)
        dump(scaler, open(scaler_save_path, 'wb'))

    X_out = []
    y_out = []
    dataset = dataset.sort_values(by=['NODE', 'ATTACK_RATIO', 'ATTACK_START_TIME', 'ATTACK_DURATION',
                                      'ATTACK_PARAMETER', 'TIME']).reset_index(drop=True)
    dataset_step = calculate_dataset_step(dataset)

    for index_start in range(0, dataset.shape[0], dataset_step):
        temp = dataset.iloc[index_start:index_start+dataset_step, :]
        temp = temp[temp_columns]
        X = temp.iloc[:,0:-num_labels]
        y = temp.iloc[:,-num_labels:]
        X = np.asarray(X).astype(float)
        y = np.asarray(y).astype(float)
        X = scaler.transform(X)

        for i in range(X.shape[0] - time_window + 1):
            X_out.append(X[i:i + time_window])
            y_out.append(y[i + time_window - 1])

    X_out, y_out = np.array(X_out), np.array(y_out)
    if nn_model == 'dense' or nn_model == 'aen':
        X_out = X_out.reshape((X_out.shape[0], X_out.shape[1]*X_out.shape[2]))
    if (data_type == 'train') and (upsample_enabled is True):
        X_out, y_out = upsample_dataset(X_out, y_out, num_labels)

    return X_out, y_out, scaler


def autoencoder(input_shape):
    '''
    Generate the neural network model
    Args:
        input_shape: the input shape of the dataset given to the model

    Returns:
        The neural network model

    '''
    #encoder
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(256, input_shape=(input_shape,), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(16, activation='relu', trainable=False))
    model.add(tf.keras.layers.BatchNormalization())

    #decoder
    model.add(tf.keras.layers.Dense(16, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dense(input_shape, activation='relu'))


    #encoder_model = encoder(input_shape)
    #autoencoder_model = decoder(input_shape, encoder_model)
    model.compile(loss='mse', optimizer='adam', metrics=[tf.keras.metrics.Accuracy()], run_eagerly=True)
    model.summary()
    return model


def setup_aen_model(nn_model, architecture, train_dataset, validation_dataset, selected_nodes_for_correlation,
                    num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled):
    train_dataset_benign = train_dataset.loc[(train_dataset['ATTACK_RATIO'] == 0)]
    validation_dataset_benign = validation_dataset.loc[(validation_dataset['ATTACK_RATIO'] == 0)]
    X_train, y_train, scaler = get_dataset_input_output(nn_model, architecture, 'train', train_dataset_benign,
                                                        selected_nodes_for_correlation, num_labels,
                                                        time_window, scaler_save_path, scaler, use_time_hour,
                                                        use_onehot, upsample_enabled)
    X_validation, y_validation, _ = get_dataset_input_output(nn_model, architecture, 'validation',
                                                             validation_dataset_benign,
                                                             selected_nodes_for_correlation, num_labels,
                                                             time_window, scaler_save_path, scaler, use_time_hour,
                                                             use_onehot, upsample_enabled)

    tf.keras.backend.clear_session()
    aen_model = autoencoder(X_train.shape[1])
    epochs_aen = 30
    batch_size_aen = 32

    history = aen_model.fit(X_train, X_train, batch_size=batch_size_aen,
                            validation_data=(X_validation, X_validation), epochs=epochs_aen,
                            verbose=1)
    return aen_model


def trans_encoder(inputs, head_size, n_heads, ff_dim, drop=0.0):
    # MultiHeadAttention
    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=n_heads, dropout=drop)(x, x)
    x = tf.keras.layers.Dropout(drop)(x)
    res = x + inputs

    x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(res)
    x = tf.keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')(x)
    x = tf.keras.layers.Dropout(drop)(x)
    x = tf.keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res


def create_nn_model(nn_model, architecture, input_shape, output_shape, l2_regularizer_val, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)

    model = tf.keras.Sequential()
    optimizer = 'adam'
    if nn_model == 'dense':
        if architecture == 'multiple_models_with_correlation' or architecture == 'one_model_with_correlation':
            l2_regularizer_val = 0.0 # dense use to have 4 neurons in previous runs
            dropout_ratio = 0.3
        else:
            l2_regularizer_val = 0.0
            dropout_ratio = 0.0
        model.add(tf.keras.layers.Dense(5, input_shape=(input_shape,), activation='relu',
                                        kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer_val)))
        model.add(tf.keras.layers.Dropout(dropout_ratio))
        model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid', bias_initializer=output_bias))
    elif nn_model == 'cnn':
        if architecture == 'multiple_models_with_correlation' or architecture == 'one_model_with_correlation':
            l2_regularizer_val = 0.0
            dropout_ratio = 0.3
        else:
            l2_regularizer_val = 0.0
            dropout_ratio = 0.0
        model.add(tf.keras.layers.Conv1D(filters=5, kernel_size=3, activation='relu', input_shape=input_shape,
                                         kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer_val)))
        model.add(tf.keras.layers.Dropout(dropout_ratio))
        model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid', bias_initializer=output_bias))
    elif nn_model == 'lstm':
        if architecture == 'multiple_models_with_correlation' or architecture == 'one_model_with_correlation':
            l2_regularizer_val = 0.3
        else:
            l2_regularizer_val = 0.0
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.LSTM(4, activation='tanh', input_shape=input_shape,
                                       kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer_val)))
        # model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid', bias_initializer=output_bias))
    elif nn_model == 'aen':
        model.add(tf.keras.layers.Dense(256, input_shape=(input_shape,), activation='relu', trainable=False))
        model.add(tf.keras.layers.BatchNormalization(trainable=False))
        model.add(tf.keras.layers.Dense(128, activation='relu', trainable=False))
        model.add(tf.keras.layers.BatchNormalization(trainable=False))
        model.add(tf.keras.layers.Dense(64, activation='relu', trainable=False))
        model.add(tf.keras.layers.BatchNormalization(trainable=False))
        model.add(tf.keras.layers.Dense(32, activation='relu', trainable=False))
        model.add(tf.keras.layers.BatchNormalization(trainable=False))
        model.add(tf.keras.layers.Dense(16, activation='relu', trainable=False))
        model.add(tf.keras.layers.BatchNormalization(trainable=False))
        model.add(tf.keras.layers.Dense(8, activation='relu'))
        model.add(tf.keras.layers.Dense(output_shape, activation='sigmoid', bias_initializer=output_bias))
    elif nn_model == 'trans':
        if architecture == 'multiple_models_with_correlation' or architecture == 'one_model_with_correlation':
            l2_regularizer_val = 0.01
        else:
            l2_regularizer_val = 0.0
        head_size = 1 # 256
        n_heads = 1
        ff_dim = 2
        n_trans_blocks = 1  # 4
        mlp_units = [2] # [128]
        drop = 0.0 # 0.25
        mlp_drop = 0.4
        inpts = tf.keras.Input(shape=input_shape)
        x = inpts
        # for _ in range(n_trans_blocks):
        #     x = trans_encoder(x, head_size, n_heads, ff_dim, drop)

        x = tf.keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=n_heads, dropout=drop,
                                               kernel_regularizer=tf.keras.regularizers.l2(l2_regularizer_val))(x, x)
        x = tf.keras.layers.GlobalAveragePooling1D(data_format='channels_first')(x)
        # for dim in mlp_units:
        #     x = tf.keras.layers.Dense(dim, activation='relu')(x)
        #     x = tf.keras.layers.Dropout(mlp_drop)(x)
        oupts = tf.keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)(x)
        model = tf.keras.Model(inpts, oupts)

    metrics = [tf.keras.metrics.Accuracy(), tf.keras.metrics.BinaryAccuracy(),
               tf.keras.metrics.BinaryCrossentropy(), tf.keras.metrics.Recall(),
               tf.keras.metrics.Precision(), tf.keras.metrics.AUC(),
               tf.keras.metrics.TruePositives(), tf.keras.metrics.TrueNegatives(),
               tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives()]

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=metrics)
    model.summary()
    return model


class ClearMemory(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        tf.keras.backend.clear_session()


def setup_callbacks(saved_model_path):
    checkpoint_path = saved_model_path + 'checkpoints/all/weights-{epoch:04d}'
    prepare_output_directory(checkpoint_path)
    cp_1 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True)

    checkpoint_path = saved_model_path + 'checkpoints/recall/weights'
    prepare_output_directory(checkpoint_path)
    cp_2 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='recall',
        mode='max',
        save_best_only=True)

    checkpoint_path = saved_model_path + 'checkpoints/val_recall/weights'
    prepare_output_directory(checkpoint_path)
    cp_3 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_recall',
        mode='max',
        save_best_only=True)


    checkpoint_path = saved_model_path + 'checkpoints/accuracy/weights'
    prepare_output_directory(checkpoint_path)
    cp_4 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='accuracy',
        mode='max',
        save_best_only=True)

    checkpoint_path = saved_model_path + 'checkpoints/val_accuracy/weights'
    prepare_output_directory(checkpoint_path)
    cp_5 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True)

    checkpoint_path = saved_model_path + 'checkpoints/loss/weights'
    prepare_output_directory(checkpoint_path)
    cp_6 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    checkpoint_path = saved_model_path + 'checkpoints/val_loss/weights'
    prepare_output_directory(checkpoint_path)
    cp_7 = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    log_path = saved_model_path + 'logs/logs.csv'
    prepare_output_directory(log_path)
    csv_logger = tf.keras.callbacks.CSVLogger(log_path, separator=',', append=False)

    # tensorboard_path = saved_model_path + 'tensorboard/' + str(datetime.now())
    # prepare_output_directory(tensorboard_path)
    # tensorboard = tf.keras.callbacks.TensorBoard(log_dir= tensorboard_path, histogram_freq=1)

    # callbacks = [cp_1, cp_2, cp_3, cp_4, cp_5, cp_6, cp_7, csv_logger, tensorboard]
    callbacks = [cp_1, cp_2, cp_3, cp_4, cp_5, cp_6, cp_7, csv_logger, ClearMemory()]
    return callbacks


def plot_logs(logs_path, output_path):
    logs = pd.read_csv(logs_path)
    metrics = logs.columns.values
    new_metrics = {}
    for metric in metrics:
        if metric[-2] == '_':
            new_metrics[metric] = metric[:-2]
        elif metric[-3] == '_':
            new_metrics[metric] = metric[:-3]

    logs = logs.rename(new_metrics, axis='columns')
    metrics = logs.columns.values

    for metric in metrics:
        if metric == 'epoch' or 'val' in metric:
            continue
        plt.clf()
        plt.plot(logs['epoch'], logs[metric], label='Train')
        # plt.plot(logs['epoch'], logs['val_'+metric], label='Validation')
        plt.xlabel('Epoch Number')
        plt.ylabel(metric)
        plt.title(metric + ' vs epoch')
        plt.legend()
        plt.savefig(output_path + metric + '.png')


def main_plot_logs(metadata_metric, nn_model, architecture, run_number):
    all_saved_models_path = CONFIG.OUTPUT_DIRECTORY + 'nn_training/' + 'metadata_' + metadata_metric + '/' +\
                        nn_model + '/' + architecture + '/' + 'run_' + str(run_number) + '/saved_model/*'
    for directory in glob.glob(all_saved_models_path):
        print(directory)
        logs_path = directory + '/logs/logs.csv'
        output_path = directory + '/logs/pics/'
        prepare_output_directory(output_path)
        plot_logs(logs_path, output_path)


def main_train_model(nn_model, architecture, use_time_hour, use_onehot, upsample_enabled, train_dataset_path,
                     validation_dataset_path, time_window, num_labels, epochs, batch_size,
                     use_metadata, metadata_path, metadata_metric, num_selected_nodes_for_correlation,
                     l2_regularizer_val, run_number):
    seed = 1
    tf.random.set_seed(seed)
    random.seed(seed)

    train_dataset_all = load_dataset(train_dataset_path)
    validation_dataset_all = load_dataset(validation_dataset_path)

    model_output_path = CONFIG.OUTPUT_DIRECTORY + 'nn_training/' + 'metadata_' + metadata_metric + '/' +\
                        nn_model + '/' + architecture + '/' + 'run_' + str(run_number) + '/saved_model/'
    prepare_output_directory(model_output_path)

    nodes = []
    if architecture == 'multiple_models_with_correlation' or architecture == 'multiple_models_without_correlation':
        nodes = list(train_dataset_all['NODE'].unique())
    elif architecture == 'one_model_with_correlation' or architecture == 'one_model_without_correlation':
        nodes = ['one_model']
    for node in nodes:
        saved_model_path = model_output_path + str(node) + '/'
        scaler_save_path = model_output_path + str(node) + '/scaler.pkl'
        prepare_output_directory(saved_model_path)

        train_dataset = pd.DataFrame()
        validation_dataset = pd.DataFrame()
        if architecture == 'multiple_models_with_correlation' or architecture == 'multiple_models_without_correlation':
            train_dataset = train_dataset_all.loc[(train_dataset_all['NODE'] == node)]
            validation_dataset = validation_dataset_all.loc[(validation_dataset_all['NODE'] == node)]
        elif architecture == 'one_model_with_correlation' or architecture == 'one_model_without_correlation':
            train_dataset = train_dataset_all
            validation_dataset = validation_dataset_all

        selected_nodes_for_correlation = list(train_dataset_all['NODE'].unique())
        if architecture == 'multiple_models_with_correlation' and use_metadata is True:
            selected_nodes_for_correlation = [node]
            if metadata_metric == 'RANDOM':
                rest_of_the_nodes = list(train_dataset_all['NODE'].unique())
                rest_of_the_nodes.remove(node)
                selected_nodes_for_correlation.extend(random.choices(rest_of_the_nodes,
                                                                     k=num_selected_nodes_for_correlation - 1))
            elif metadata_metric == 'SHAP':
                metadata_dataset_all = load_dataset(metadata_path)
                metadata_dataset = metadata_dataset_all.loc[(metadata_dataset_all['node'] == node)]
                selected_nodes_for_correlation = list(metadata_dataset['feature_node'][0:num_selected_nodes_for_correlation].values)
            else:
                metadata_dataset_all = load_dataset(metadata_path)
                metadata_dataset = metadata_dataset_all.loc[(metadata_dataset_all['NODE_1'] == node)]
                metadata_dataset = metadata_dataset.sort_values(by=[metadata_metric])
                selected_nodes_for_correlation.extend(list(metadata_dataset['NODE_2'][0:num_selected_nodes_for_correlation-1].values))
        selected_nodes_save_path = model_output_path + str(node) + '/selected_nodes_for_correlation.pkl'
        dump(selected_nodes_for_correlation, open(selected_nodes_save_path, 'wb'))

        scaler = StandardScaler()
        X_train, y_train, scaler = get_dataset_input_output(nn_model, architecture, 'train', train_dataset,
                                                            selected_nodes_for_correlation, num_labels,
                                                            time_window, scaler_save_path, scaler, use_time_hour,
                                                            use_onehot, upsample_enabled)
        X_validation, y_validation, _ = get_dataset_input_output(nn_model, architecture, 'validation', validation_dataset,
                                                                 selected_nodes_for_correlation, num_labels,
                                                                 time_window, scaler_save_path, scaler, use_time_hour,
                                                                 use_onehot, upsample_enabled)
        input_shape = 0
        output_shape = 0
        if nn_model == 'dense' or nn_model == 'aen':
            input_shape = X_train.shape[1]
            output_shape = y_train.shape[1]
        elif nn_model == 'cnn' or nn_model == 'lstm' or nn_model == 'trans':
            input_shape = (X_train.shape[1], X_train.shape[2])
            output_shape = y_train.shape[1]


        neg, pos = np.bincount(train_dataset['ATTACKED'])
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)
        class_weight = {0: weight_for_0, 1: weight_for_1}
        initial_bias = np.log([pos / neg])

        if nn_model == 'aen':
            aen_model = setup_aen_model(nn_model, architecture, train_dataset, validation_dataset, selected_nodes_for_correlation,
                            num_labels, time_window, scaler_save_path, scaler, use_time_hour, use_onehot, upsample_enabled)

        tf.keras.backend.clear_session()
        model = create_nn_model(nn_model, architecture,input_shape, output_shape, l2_regularizer_val, initial_bias)
        if nn_model == 'aen':
            for l1, l2 in zip(model.layers[0:10], aen_model.layers[0:10]):
                l1.set_weights(l2.get_weights())

        callbacks_list = setup_callbacks(saved_model_path)
        # print(X_train.shape)
        # print(y_train.shape)
        # print(X_validation.shape)
        # print(y_validation.shape)
        model.fit(X_train, y_train, batch_size=batch_size, # validation_data=(X_validation, y_validation),
                  epochs=epochs, verbose=1, callbacks=callbacks_list, class_weight=class_weight)
        model.save(saved_model_path + 'final_model')


def main():
    use_time_hour = False
    use_onehot = True
    upsample_enabled = False
    epochs = 3
    batch_size = 32
    num_labels = 1
    time_window = 10
    train_dataset_path = CONFIG.OUTPUT_DIRECTORY + 'pre_process/Output/train_data/train_data.csv'
    validation_dataset_path = CONFIG.OUTPUT_DIRECTORY + 'pre_process/Output/validation_data/validation_data.csv'
    print(sys.argv)
    l2_regularizer_val = 0

    if len(sys.argv) > 1:
        nn_model = sys.argv[1]
        architecture = sys.argv[2]
        use_metadata = sys.argv[3] == 'True'
        metadata_path = sys.argv[4]
        metadata_metric = sys.argv[5]
        num_selected_nodes_for_correlation = int(sys.argv[6])
        run_number = int(sys.argv[7])

        main_train_model(nn_model, architecture, use_time_hour, use_onehot, upsample_enabled, train_dataset_path,
                         validation_dataset_path, time_window, num_labels, epochs, batch_size,
                         use_metadata, metadata_path, metadata_metric, num_selected_nodes_for_correlation,
                         l2_regularizer_val, run_number)
        main_plot_logs(metadata_metric, nn_model, architecture, run_number)
    else:
        nn_model_list = ['dense', 'cnn', 'lstm', 'trans', 'aen']
        architecture_list = ['multiple_models_with_correlation',
                             'multiple_models_without_correlation',
                             'one_model_with_correlation',
                             'one_model_without_correlation']
        nn_model_list = ['aen']
        architecture_list = ['multiple_models_with_correlation']

        # Run 1
        use_metadata = False
        metadata_path = CONFIG.OUTPUT_DIRECTORY
        metadata_metric = 'NOT_USED'
        num_selected_nodes_for_correlation = 0
        run_number = 0
        for nn_model in nn_model_list:
            for architecture in architecture_list:
                main_train_model(nn_model, architecture, use_time_hour, use_onehot, upsample_enabled, train_dataset_path,
                                 validation_dataset_path, time_window, num_labels, epochs, batch_size,
                                 use_metadata, metadata_path, metadata_metric, num_selected_nodes_for_correlation,
                                 l2_regularizer_val, run_number)
                main_plot_logs(metadata_metric, nn_model, architecture, run_number)

        # Run 2
        nn_model_list = ['lstm']
        architecture_list = ['multiple_models_with_correlation']
        use_metadata = True
        metadata_path_list = [CONFIG.OUTPUT_DIRECTORY + 'pre_process/Output/metadata/distances.csv',
                              CONFIG.OUTPUT_DIRECTORY + 'pre_process/Output/metadata/correlations.csv',
                              '']
        metadata_metric_list = ['DISTANCE', 'CORRELATION', 'RANDOM']
        # metadata_path_list = [CONFIG.OUTPUT_DIRECTORY + 'pre_process/Output/metadata/feature_importance.csv']
        # metadata_metric_list = ['SHAP']
        num_random_trains = 10
        num_selected_nodes_for_correlation = 5
        for metadata_index in range(3):
            metadata_path = metadata_path_list[metadata_index]
            metadata_metric = metadata_metric_list[metadata_index]
            for nn_model in nn_model_list:
                for architecture in architecture_list:
                    if metadata_metric != 'RANDOM':
                        run_number = 0
                        main_train_model(nn_model, architecture, use_time_hour, use_onehot, upsample_enabled, train_dataset_path,
                                         validation_dataset_path, time_window, num_labels, epochs, batch_size,
                                         use_metadata, metadata_path, metadata_metric,
                                         num_selected_nodes_for_correlation, l2_regularizer_val, run_number)
                        main_plot_logs(metadata_metric, nn_model, architecture, run_number)
                    else:
                        for run_number in range(num_random_trains):
                            main_train_model(nn_model, architecture, use_time_hour, use_onehot, upsample_enabled, train_dataset_path,
                                             validation_dataset_path, time_window, num_labels, epochs, batch_size,
                                             use_metadata, metadata_path, metadata_metric,
                                             num_selected_nodes_for_correlation, l2_regularizer_val, run_number)
                            main_plot_logs(metadata_metric, nn_model, architecture, run_number)


if __name__ == '__main__':
    main()
