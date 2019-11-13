from keras.models import Sequential
from keras.layers import Dense
import numpy

from keras.utils import np_utils
from keras import backend as K
import pandas as pd
import random
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
TrainDataset = pd.read_csv("./train.csv")
TestDataset = pd.read_csv("./test.csv")
y = TrainDataset.iloc[1:,1].values.ravel()
X = TrainDataset.iloc[1:,2:].values
X_test = TestDataset.iloc[1:,1:].values

# split data into train = 0.8*number of dataset, validate = 0.2*number of dataset
nRows = X.shape[0]
nColumns = X.shape[1]
validRatio = 0.25
nValid = int(validRatio*nRows)
nTrain = nRows - nValid

validIndex = random.sample(range(nRows), nValid)
trainIndex = numpy.array(list(set(numpy.arange(0,nRows)) - set(validIndex)))
X_valid = X[validIndex,]
X_train = X[trainIndex,]
y_valid = y[validIndex,]
y_train = y[trainIndex,]

# normalize inputs from 0-255 to 0-1
#X_train = X_train / 255
#X_test = X_test / 255
# one hot encode outputs
#y_train = np_utils.to_categorical(y_train)
#y_valid = np_utils.to_categorical(y_valid)
num_classes = 99

def nn_model(num_classes):
    # create model
    model = Sequential()
    model.add(Dense(24, input_dim=8, init='uniform', activation='relu'))
    model.add(Dense(48, init='uniform', activation='relu'))
    model.add(Dense(24, init='uniform', activation='relu'))
    model.add(Dense(num_classes, init='uniform', activation='sigmoid'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    
    # build the model
model = nn_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), nb_epoch=50, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))

y_test = model.predict_classes(X_test, verbose=0)
numpy.savetxt('mnist-dada.csv', numpy.c_[range(1,len(y_test)+1),y_test], delimiter=',', header = 'ImageId,Label', comments = '', fmt='%d')