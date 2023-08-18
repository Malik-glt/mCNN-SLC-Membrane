#import libraries 
import os
import csv
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf

# Initialization
MAX_SEQ_LENGTH = 2560    # Maximum sequence length utilized in this study 
NUM_FEATURE = 1024       # number of features per residue
BATCH_SIZE = 4
NUM_CLASSES = 2
EPOCHS = 100

#Generator Function For Prottrans embeddings
import pandas as pd
import os
import numpy as np

def data_generator(folder_path, labels_path, batch_size=BATCH_SIZE, max_seq_length=MAX_SEQ_LENGTH):
    labels_df = pd.read_csv(labels_path)
    labels_dict = dict(zip(labels_df['ID'], labels_df['Label']))

    data = []
    labels = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.porttrans'):
            with open(os.path.join(folder_path, filename), 'r') as f:
                porttrans = np.loadtxt(f)

            protein_id = filename[:-10]
            label = labels_dict.get(protein_id, None)

            if label is not None:
                if porttrans.shape[0] < max_seq_length:
                    porttrans = np.pad(porttrans, ((0, max_seq_length - porttrans.shape[0]), (0, 0)), mode='constant')
                else:
                    porttrans = porttrans[:max_seq_length, :]

                data.append(porttrans)
                labels.append(label)

            if len(data) >= batch_size:
                yield np.array(data).reshape(-1, 1, max_seq_length, NUM_FEATURE), tf.keras.utils.to_categorical(np.array(labels), NUM_CLASSES)
                data = []
                labels = []

    if data:
        yield np.array(data).reshape(-1, 1, max_seq_length, NUM_FEATURE), tf.keras.utils.to_categorical(np.array(labels), NUM_CLASSES)

# Build Model for diffrent window sizes 
import os
import tensorflow as tf
import math
from sklearn import metrics
from sklearn.metrics import roc_curve
from tensorflow.keras import Model, layers

class DeepScan(Model):
    def __init__(self,
                 input_shape=(1, MAX_SEQ_LENGTH, NUM_FEATURE),
                 window_sizes=[2, 4, 6, 8, 10],
                 num_filters=128,
                 num_hidden=512):
        super(DeepScan, self).__init__()
        # Add input layer
        self.input_layer = tf.keras.Input(input_shape)
        self.window_sizes = window_sizes
        self.conv2d = []
        self.maxpool = []
        self.flatten = []
        for window_size in self.window_sizes:
            self.conv2d.append(layers.Conv2D(
                filters=num_filters,
                kernel_size=(1, window_size),
                activation=tf.nn.relu,
                padding='valid',
                bias_initializer=tf.constant_initializer(0.1),
                kernel_initializer=tf.keras.initializers.GlorotUniform()
            ))
            self.maxpool.append(layers.MaxPooling2D(
                pool_size=(1, MAX_SEQ_LENGTH - window_size + 1),
                strides=(1, MAX_SEQ_LENGTH),
                padding='valid'))
            self.flatten.append(layers.Flatten())
        self.dropout = layers.Dropout(rate=0.7)
        self.fc1 = layers.Dense(
            num_hidden,
            activation=tf.nn.relu,
            bias_initializer=tf.constant_initializer(0.1),
            kernel_initializer=tf.keras.initializers.GlorotUniform()
        )
        self.fc2 = layers.Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(1e-3))

        # Get output layer with `call` method
        self.out = self.call(self.input_layer)

    def call(self, x, training=False):
        _x = []
        for i in range(len(self.window_sizes)):
            x_conv = self.conv2d[i](x)
            x_maxp = self.maxpool[i](x_conv)
            x_flat = self.flatten[i](x_maxp)
            _x.append(x_flat)

        x = tf.concat(_x, 1)
        x = self.dropout(x, training=training)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

# Train the Model For the sequence lenght=2560
NUM_FILTER = 128
NUM_HIDDEN = 512
WINDOW_SIZES = [2, 4, 6, 8, 10]
LOG_DIR = f'Meb_Sec_f{NUM_FILTER}_h{NUM_HIDDEN}_{"-".join([str(int) for int in WINDOW_SIZES])}'
os.makedirs(LOG_DIR, exist_ok=True)

model = DeepScan(
    num_filters=NUM_FILTER,
    num_hidden=NUM_HIDDEN,
    window_sizes=WINDOW_SIZES
)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.build(input_shape=(BATCH_SIZE, 1, MAX_SEQ_LENGTH, NUM_FEATURE))
model.summary()

# Calculate the number of steps per epoch
total_train_samples = 4568  # Update with the correct total number of training samples
total_test_samples = 1142 # Update with the correct total number of testing samples
train_steps_per_epoch = int(np.ceil(total_train_samples / BATCH_SIZE))
test_steps_per_epoch = int(np.ceil(total_test_samples / BATCH_SIZE))

# Define the validation callback function
def val_binary(epoch, logs):
    try:
        # Recreate the test generator for each validation step to reset it
        test_generator = data_generator('pathe_Testing_Data', 'path_Testing_Labels')

        # Predict on test data using the test generator
        pred = model.predict(test_generator)


    except StopIteration:
        pass

# Train the model for multiple epochs
for epoch in range(EPOCHS):
    # Create a new data generator for each epoch to ensure random shuffling
    train_generator = data_generator('pathe_Traning_Data', 'path_Traning_Labels')
    test_generator = data_generator('pathe_Testing_Data', 'path_Testing_Labels')

    model.fit(
        train_generator,
        steps_per_epoch=train_steps_per_epoch,
        epochs=1,  # Train for one epoch at a time
        validation_data=test_generator,
        validation_steps=test_steps_per_epoch,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_epoch_end=val_binary),
            tf.keras.callbacks.ModelCheckpoint(LOG_DIR + '/weights.{epoch:02d}', save_weights_only=True),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        ]
    )



# Define the prediction callback function.
from sklearn.metrics import auc
from sklearn.metrics import auc, RocCurveDisplay
from sklearn.metrics import auc, RocCurveDisplay, confusion_matrix

def pred_binary(epoch, logs):
    try:
        # Recreate the test generator for each prediction step to reset it
        test_generator = data_generator('pathe_Testing_Data', 'path_Testing_Labels')

        # Initialize an empty list for storing the true labels
        y_test_labels = []
        pred_test = None

        # Iterate over the test data generator to extract the true labels and predict iteratively
        while True:
            try:
                _, labels = next(test_generator)
                # Extract the labels for the current batch
                batch_labels = np.argmax(labels, axis=1)
                # Append the batch labels to the list
                y_test_labels.extend(batch_labels)

                # Predict on the current batch using the trained model
                batch_pred = model.predict_on_batch(_)

                # Accumulate predictions for each batch
                if pred_test is None:
                    pred_test = batch_pred
                else:
                    pred_test = np.concatenate((pred_test, batch_pred))

            except StopIteration:
                break

        # Convert the list of labels to a numpy array
        y_test_categorical = np.array(y_test_labels)

        # Extract the probabilities for positive class
        y_pred_prob = pred_test[:, 1]

        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_test_categorical, y_pred_prob)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='mCNN-SLC-Membrane')
        display.plot()

        # Find the best threshold based on g-mean
        gmeans = np.sqrt(tpr * (1 - fpr))
        best_threshold_idx = np.argmax(gmeans)
        best_threshold = thresholds[best_threshold_idx]

        # Convert probabilities to binary predictions based on the best threshold
        y_pred = (y_pred_prob >= best_threshold).astype(int)

        # Calculate evaluation metrics
        TN, FP, FN, TP = confusion_matrix(y_test_categorical, y_pred).ravel()

        Sens = TP / (TP + FN) if TP + FN > 0 else 0.0
        Spec = TN / (FP + TN) if FP + TN > 0 else 0.0
        Acc = (TP + TN) / (TP + FP + TN + FN)
        MCC = (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)) if TP + FP > 0 and FP + TN > 0 and TP + FN and TN + FN else 0.0
        F1 = 2 * TP / (2 * TP + FP + FN)
        #balanced_acc = balanced_accuracy_score(y_test_labels, y_pred_labels)
        print(f'TP={TP}, FP={FP}, TN={TN}, FN={FN}, Sens={Sens:.4f}, Spec={Spec:.4f}, Acc={Acc:.4f}, MCC={MCC:.4f}, F1={F1:.4f}')
        
    except StopIteration:
        print("StopIteration error occurred. Adjust your test data or generator.")

# Perform predictions
pred_binary(epoch=0, logs={})
