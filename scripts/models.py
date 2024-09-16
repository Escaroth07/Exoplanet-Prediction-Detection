import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

def FCN_model(len_seq):
   
    model = tf.keras.Sequential()
    
    # First Conv1D layer
    model.add(layers.Conv1D(filters=256, kernel_size=8, activation='relu', 
                            padding='same', input_shape=(len_seq, 1)))
    model.add(layers.MaxPooling1D(pool_size=5))
    model.add(layers.BatchNormalization())
    
    # Second Conv1D layer
    model.add(layers.Conv1D(filters=340, kernel_size=6, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=5))
    model.add(layers.BatchNormalization())
    
    # Third Conv1D layer
    model.add(layers.Conv1D(filters=256, kernel_size=4, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=5))
    model.add(layers.BatchNormalization())
    
    # Flatten and Dropout layers
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.3))
    
    # Dense layers
    model.add(layers.Dense(24, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(12, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    
    return model

def SVC_model(C=1.0, gamma='scale', kernel='linear'):
    model = SVC(C=C, gamma=gamma, kernel=kernel)
    return model