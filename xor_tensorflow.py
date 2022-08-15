# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
# Source: https://github.com/bayrell-tutorials/perceptron-xor
# Link:
# https://blog.bayrell.org/ru/iskusstvennyj-intellekt/411-obuchenie-mnogoslojnogo-perseptrona-operaczii-xor.html
##

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input


# Step 1. Prepare DataSet
data_train = [
	{ "in": [0, 0], "out": [0] },
	{ "in": [0, 1], "out": [1] },
	{ "in": [1, 0], "out": [1] },
	{ "in": [1, 1], "out": [0] },
]


# Convert to question and answer DataSet
data_train_question = list(map(lambda item: item["in"], data_train))
data_train_answer = list(map(lambda item: item["out"], data_train))


# Normalize
data_train_question = np.array(data_train_question, "float32")
data_train_answer = np.array(data_train_answer, "float32")


# Print info
print ("Input:")
print (data_train_question)
print ("Shape:", data_train_question.shape)
print ("")
print ("Answers:")
print (data_train_answer)
print ("Shape:", data_train_answer.shape)

# Wait
#print ("Press Enter to continue")
#input()


# Step 2. Create tensorflow model
model = Sequential(name='XOR_Model')
model.add(Input(shape=(2), name='input'))
model.add(Dense(16, name='hidden', activation='relu'))
model.add(Dense(1, name='output', activation='sigmoid'))

# Compile
model.compile(loss='mean_squared_error', 
              optimizer='adam',
              metrics=['accuracy'])
		
# Output model info to the screen		
model.summary()

# Wait
#print ("Press Enter to continue")
#input()


# Step 3. Train model
history = model.fit(
	data_train_question, # Input
	data_train_answer,   # Output
	batch_size=4,
	epochs=250,
	verbose=1)

plt.plot( np.multiply(history.history['accuracy'], 100), label='Correct answers')
plt.plot( np.multiply(history.history['loss'], 100), label='Error')
plt.ylabel('%')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('xor_tensorflow_model.png')
plt.show()


# Step 3. Test model
test = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
]

test = np.asarray(test)

print ("Shape:", test.shape)

answer = model.predict( test )

for i in range(0,len(answer)):
  print(test[i], "->", answer[i].round())
