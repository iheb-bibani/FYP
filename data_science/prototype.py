import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from postgres import connection
import psycopg2
from datetime import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import autokeras as ak

def main():
    all_data, null_volumes = download_data(connection)
    
    
    for stock, df in all_data.items():
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df['Future_Quarter_Return'] = df['Adj_Close'].pct_change(periods=63)
        df['Future_Quarter_Return'] = df['Future_Quarter_Return'].shift(-63)
        df.dropna(subset=['Future_Quarter_Return'], inplace=True)
        all_data[stock] = df
        print(df)
    
    features = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Future_Quarter_Return']    
    model = stacked_rnn_model_multi(features)

    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    all_data_concat = pd.concat(all_data.values())
    data_to_predict = np.array(all_data_concat[features])
    length = len(data_to_predict)
    time_steps = np.array(range(length)) * 3 
    window_size = 63
    
    scaler = StandardScaler()
    rescaler = StandardScaler()
    #scaler = MinMaxScaler(feature_range=(0, 1))
    data_to_predict_normalizer = scaler.fit_transform(data_to_predict)
    adj_close = all_data_concat['Future_Quarter_Return'].values.reshape(-1, 1)
    adj_close = rescaler.fit_transform(adj_close)
    train_adj_close, train_time_steps, test_adj_close, test_time_steps = split_data(data_to_predict_normalizer, time_steps, 0.8)
    train_adj_close, train_time_steps, val_adj_close, val_time_steps = split_data(train_adj_close, train_time_steps, 0.9)

    train_data = wrangle_data(train_adj_close, 'train', window_size, 32)
    val_data = wrangle_data(val_adj_close, 'val', window_size, 32)
    test_data = wrangle_data(test_adj_close, 'test', window_size, 32)

    with tf.device("/GPU:0"):
        history = model.fit(train_data, epochs=20, validation_data=val_data, callbacks=[early_stop], use_multiprocessing=True, workers=12)
    plot_history(history)
    model.evaluate(test_data)

    # Adjust prediction and plotting to handle multiple features
    predictions = model.predict(test_data)
    predictions = predictions[:,-1].reshape(-1, 1)
    predictions = rescaler.inverse_transform(predictions)
    test_volume_true = test_adj_close[window_size:,][:,0].reshape(-1, 1)
    test_volume_true = rescaler.inverse_transform(test_volume_true) 
    predictions = predictions.reshape(-1)
    test_volume_true = test_volume_true.reshape(-1)
    show_predictions(model, test_data, test_volume_true, test_time_steps[window_size:], end=2*1)


    
def download_data(connection: psycopg2.extensions.connection, start_date = "2010-01-04", end_date = "2023-03-30") -> pd.DataFrame:
    cursor = connection.cursor()
    cursor.execute("SELECT ticker FROM equities")
    tickers = cursor.fetchall()
    stock_data = {}
    null_volumes = {}

    for stock_tuple in tickers:
        stock = stock_tuple[0]
        if stock == "^STI":
            continue
        table_name = f"stock_{stock[:3]}"  # Remove the .SI suffix
        query = (
            f"SELECT * FROM {table_name} ORDER BY Date ASC" # WHERE Date BETWEEN %s AND %s
        )
        cursor.execute(query, (start_date, end_date))
        data = cursor.fetchall()
        if data:
            dates, open_price, high, low, close, adj_close, volume = zip(*data)
            
            # if datetime.strptime(start_date, "%Y-%m-%d") != datetime.strptime(dates[0], "%Y-%m-%d") \
            #     or datetime.strptime(end_date, "%Y-%m-%d") != datetime.strptime(dates[-1], "%Y-%m-%d"):
            #     print(f"Stock {stock} only has data from {dates[0]} to {dates[-1]}")
            #     continue

            df = pd.DataFrame({
                'Date': pd.to_datetime(dates),
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Adj_Close': adj_close,
                'Volume': volume
            })
            #if df[-1:]['Adj_Close'].values[0] < 0.2: continue
            # Handle special case where all columns are equal
            mask = (df['Open'] == df['High']) & (df['High'] == df['Low']) & (df['Low'] == df['Close']) & (df['Close'] == df['Adj_Close'])
            df.loc[mask, 'Volume'] = -1.0
            
            df.replace(0, np.nan, inplace=True)
            
            null_counts = df.isnull().sum()
            if null_counts.any():
                null_volumes[stock] = {'null_count': null_counts['Volume'], '0_vol': df[df['Volume'] == -1.0].shape[0]}  
            stock_data[stock] = df

    cursor.close()
    return stock_data, null_volumes

def compile_model(new_model, loss='mean_squared_error'):
    #adam = tf.keras.optimizers.Adam(learning_rate=0.02)
    new_model.compile(optimizer='adam', loss=loss, metrics=['mean_absolute_error'])
    print(new_model.summary())
    return new_model

def stacked_rnn_model_multi(features):
    tf.keras.backend.clear_session()
    new_model = tf.keras.Sequential([
        tf.keras.layers.InputLayer((None, len(features))),
        tf.keras.layers.Conv1D(30, kernel_size=6, padding='causal', activation='relu'),
        tf.keras.layers.LSTM(60, return_sequences=True),
        tf.keras.layers.LSTM(60),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    return compile_model(new_model)

def plot_history(history, metrics=None):

    if isinstance(metrics, str):
        metrics = [metrics]
    if metrics is None:
        metrics = [x for x in history.history.keys() if x[:4] != 'val_']
    if len(metrics) == 0:
        print('No metrics to plot')
        return
    
    x = history.epoch
    
    rows = 1
    cols = len(metrics)
    count = 0
    
    plt.figure(figsize=(12* cols, 8))
    
    for metric in sorted(metrics):
        count+=1
        plt.subplot(rows, cols, count)
        plt.plot(x, history.history[metric], label='Train')
        val_metric = f'val_{metric}'
        if val_metric in history.history.keys():
            plt.plot(x, history.history[val_metric], label='Validation')
        plt.title(metric.capitalize())
        plt.legend()
        
    plt.show()

def plot_sequence(time, sequences, start=0, end=None):
    y_max = 1.0
    
    if len(np.shape(sequences)) == 1:
        print('Initial Shape', np.shape(sequences))
        sequences = [sequences]
        print('Shape: ', np.shape(sequences))
    
    time = time[start:end]
    plt.figure(figsize=(12, 8))
    
    for sequence in sequences:
        print('Sequence shape: ', np.shape(sequence))
        y_max = max(y_max, np.max(sequence))
        sequence = sequence[start:end]
        plt.plot(time, sequence)
        
    plt.ylim(-2, y_max)
    plt.xlim(time[start], time[-1])
    plt.show()


def show_predictions(trained_model, predict_sequence, true_values, predict_time, begin=0, end=None):
    predictions = trained_model.predict(predict_sequence)
    print(predictions.shape)
    predictions = predictions[:,-1].reshape(len(predictions))
    print(predictions.shape)
    plot_sequence(predict_time, (true_values, predictions), begin, end)
    return predictions

def split_data(data, time, split_size):
    d_split = int(np.ceil(len(data) * split_size))
    
    big_data = data[:d_split]
    big_time = time[:d_split]
    small_data = data[d_split:]
    small_time = time[d_split:]
    
    return big_data, big_time, small_data, small_time

@tf.autograph.experimental.do_not_convert
def wrangle_data(sequence, data_split, examples, batch_size):
    
    # Add extra data point for labels
    examples += 1
    
    # Add a rank for the data points
    # Current: Rank 1 with 100 data points -> Rank 2 with 100 examples of 1 data point
    print('Sequence shape: ', np.shape(sequence))
    seq_expand = tf.expand_dims(sequence, -1)
    print('Sequence shape: ', np.shape(seq_expand))

    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices(seq_expand)
    # Window and shift the data
    dataset = dataset.window(examples, shift=1, drop_remainder=True)
    # Convert from dataset of datasets to dataset of tensors
    dataset = dataset.flat_map(lambda x: x.batch(examples))
    
    dataset = dataset.map(lambda x: (x[:-1], x[-1]))
    
    if data_split == 'train':
        dataset = dataset.shuffle(1000)
    else:
        dataset = dataset.cache()
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset

if __name__ == "__main__":
    main()
