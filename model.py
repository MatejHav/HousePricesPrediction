import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.losses import *

from get_data import *

number_of_features = 10
train, test, train_label, test_label = split_data(get_data('data/train.csv'))
train = train.fillna(train.mean())
test = test.fillna(train.mean())
train, selected_features = pipeline(train, train_label, number_of_features)
test = test[selected_features]
final_check = get_data('data/test.csv', selected_features)
final_check = final_check.fillna(train.mean())
ids = get_data('data/test.csv', ['Id'])

# Normalize all data
normalizer = layers.Normalization(input_shape=[number_of_features,])
normalizer.adapt(train.to_numpy())


# Define the shape of the model
model = keras.Sequential([normalizer, layers.Dense(units=1)])

# Summary
print(model.summary())

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.5),
    loss=MeanSquaredLogarithmicError())

# Fit the model
statistics = model.fit(
    train,
    train_label,
    epochs=100,
    batch_size=30,
    verbose=1,
    validation_split=0.2
)

# Compute our score
predictions = model.predict(test)
score = keras.losses.mean_squared_logarithmic_error(test_label, predictions).numpy().mean()
print(f'Our score: {score}')

# Predict the test.csv and create a submission
final_predictions = model.predict(final_check).flatten()
submission = pd.DataFrame({'Id': ids['Id'], 'SalePrice': final_predictions})
submission.to_csv("data/submission.csv", index=False)
