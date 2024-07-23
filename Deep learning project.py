import pandas as pd
from collections import Counter
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import  InputLayer
from tensorflow.keras.layers import  Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
sys.stdout.reconfigure(encoding='utf-8')

#read csv
data = pd.read_csv('cover_data.csv')
#print(data.head())
#print(data['class'].describe())

#data preparation
y = data['class']
x = data.iloc[:,0:-1]
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=12, train_size = 0.8, test_size=0.2)
y_train = tensorflow.keras.utils.to_categorical(y_train)
y_test = tensorflow.keras.utils.to_categorical(y_test)
# normalize data
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

#deep learning model set up
model = Sequential()
model.add(InputLayer(input_shape=(x_train.shape[1],)))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=4, mode='min', verbose=1)

model.fit(x_train, y_train,validation_data = (x_test, y_test), epochs = 100, batch_size = 1024, verbose = 1,callbacks=[early_stopping])

y_estimate = model.predict(x_test)
y_estimate = np.argmax(y_estimate, axis = 1)
y_true = np.argmax(y_test, axis = 1)
print(classification_report(y_true, y_estimate))

#plot confusion matrix
cm = confusion_matrix(y_true, y_estimate)
labels = ['Spruce/Fir', 'Lodgeploe Pine', 'Ponderosa Pine' ,'Cottonwood/Willow', 'Aspen', 'Douglas-fir', 'Krummholz']
 
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)

#add percentage annotations
for i in range(len(labels)):
    for j in range(len(labels)):
        text = f"{cm[i, j]} ({cm[i, j] / cm[i].sum() * 100:.1f}%)"
        plt.text(j + 0.5, i + 0.5, text, ha='center', va='center', color="white" if cm[i, j] > 50000 else "black")
       
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()