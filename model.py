import tensorflow as tf
from keras import layers, models

def build_model(input_shape):
    model = models.Sequential([
        layers.Dense(32, activation=\'relu\', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation=\'relu\'),
        layers.Dropout(0.3),
        layers.Dense(16, activation=\'relu\'),
        layers.Dropout(0.3),
        layers.Dense(1, activation=\'sigmoid\')
    ])
    model.compile(optimizer=\'adam\',
                  loss=\'binary_crossentropy\',
                  metrics=[\'accuracy\'])
    return model

if __name__ == "__main__":
    # Example usage: Assuming input_shape is 19 based on previous analysis
    sample_input_shape = 19
    model = build_model(sample_input_shape)
    model.summary()
