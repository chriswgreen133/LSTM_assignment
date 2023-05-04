import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_absolute_error

# Text to CSV
df = pd.read_csv(r'household_power_consumption.txt', sep=';', header=0, low_memory=False, infer_datetime_format=True, parse_dates={'datetime':[0,1]}, index_col=['datetime'])
print(df.head())

# Imputing Null values
df = df.replace('?', np.nan)
df.isnull().sum()


def fill_missing(values):
    one_day = 60*24
    for row in range(df.shape[0]):
        for col in range(df.shape[1]):
            if np.isnan(values[row][col]):
                values[row, col] = values[row-one_day, col]


df = df.astype('float32')
fill_missing(df.values)
df.isnull().sum()

# Down sampling of Data from minutes to Days
daily_df = df.resample('D').sum()
print(daily_df.head())
print(daily_df.shape, df.shape)
# For this case, let's assume that
# Given past 10 days observation, forecast the next 5 days observations.
n_past = 10
n_future = 5
n_features = 7

# Train - Test Split
train_df,test_df = daily_df[1:1081], daily_df[1081:]  # 75% and 25%
print(train_df.shape, test_df.shape)


# Scaling the values for faster training of the models.
train = train_df
scalers = {}

for i in train_df.columns:
    scaler = MinMaxScaler(feature_range=(-1, 1))
    s_s = scaler.fit_transform(train[i].values.reshape(-1, 1))
    s_s = np.reshape(s_s, len(s_s))
    scalers['scaler_' + i] = scaler
    train[i] = s_s

test = test_df
for i in train_df.columns:
    scaler = scalers['scaler_'+i]
    s_s = scaler.transform(test[i].values.reshape(-1, 1))
    s_s = np.reshape(s_s , len(s_s))
    scalers['scaler_'+i] = scaler
    test[i] = s_s


# Converting the series to samples for supervised learning
def split_series(series, n_past, n_future):
  # n_past ==> no of past observations
  # n_future ==> no of future observations
  X, y = list(), list()
  for window_start in range(len(series)):
    past_end = window_start + n_past
    future_end = past_end + n_future
    if future_end > len(series):
      break
    # slicing the past and future parts of the window
    past, future = series[window_start:past_end, :], series[past_end:future_end, :]
    X.append(past)
    y.append(future)
  return np.array(X), np.array(y)


X_train, y_train = split_series(train.values, n_past, n_future)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1],n_features))
y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], n_features))

X_test, y_test = split_series(test.values,n_past, n_future)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1],n_features))
y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], n_features))
print(X_test.shape)

# Model Architecture
# Sequence to Sequence Model with two encoder layers and two decoder layers.
# n_features ==> no of features at each timestep in the data.
# Sequence to Sequence Model with three hidden LSTM layers and two different activation functions.

encoder_inputs = tf.keras.layers.Input(shape=(n_past, n_features))

# First LSTM layer
encoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True, activation='tanh')
encoder_outputs1, state_h1, state_c1 = encoder_l1(encoder_inputs)

# Second LSTM layer
encoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True, return_state=True, activation='tanh')
encoder_outputs2, state_h2, state_c2 = encoder_l2(encoder_outputs1)

# Third LSTM layer
encoder_l3 = tf.keras.layers.LSTM(100, return_state=True, activation='relu')
encoder_outputs3, state_h3, state_c3 = encoder_l3(encoder_outputs2)

# RepeatVector for the decoder
decoder_inputs = tf.keras.layers.RepeatVector(n_future)(encoder_outputs3)

# Decoder with LSTM layers and respective activation functions
decoder_l1 = tf.keras.layers.LSTM(100, return_sequences=True, activation='tanh')(decoder_inputs, initial_state=[state_h1, state_c1])
decoder_l2 = tf.keras.layers.LSTM(100, return_sequences=True, activation='tanh')(decoder_l1, initial_state=[state_h2, state_c2])
decoder_l3 = tf.keras.layers.LSTM(100, return_sequences=True, activation='relu')(decoder_l2, initial_state=[state_h3, state_c3])

# TimeDistributed Dense layer
decoder_outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(n_features))(decoder_l3)

# Create the model
model = tf.keras.models.Model(encoder_inputs, decoder_outputs)

print(model.summary())

# Training the models
reduce_lr = tf.keras.callbacks.LearningRateScheduler(lambda x: 1e-3 * 0.90 ** x)

# def custom_loss(y_true, y_pred):
#     y_true_repeated = tf.tile(y_true[:, -1:, :], [1, 5, 1])
#     return tf.keras.losses.Huber()(y_true_repeated, y_pred)
#
# model.compile(loss=custom_loss, optimizer='adam')

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.Huber())
history = model.fit(X_train, y_train, epochs=25, validation_data=(X_test, y_test), batch_size=32, verbose=0, callbacks=[reduce_lr])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title("E2D2 Model Loss")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Valid'])
plt.show()

# Prediction on validation set
pred1 = model.predict(X_test)
pred = model.predict(X_train)

# Inverse Scaling of the predicted values
for index, i in enumerate(train_df.columns):
    scaler = scalers['scaler_' + i]

    pred1[:, :, index] = scaler.inverse_transform(pred1[:, :, index])
    pred[:, :, index] = scaler.inverse_transform(pred[:, :, index])

    y_train[:, :, index] = scaler.inverse_transform(y_train[:, :, index])
    y_test[:, :, index] = scaler.inverse_transform(y_test[:, :, index])

# Checking Error
# Predicting Values for 3 days
for index, i in enumerate(train_df.columns):
    print(i)
    for j in range(1, 4):
        print("Day ", j, ":")
        print('Predicted Value: ', pred1[:, j - 1, index])
        print("MAE : ", mean_absolute_error(y_test[:, j - 1, index], pred1[:, j - 1, index]))
    print()
    print()

