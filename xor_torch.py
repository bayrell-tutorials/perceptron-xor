# -*- coding: utf-8 -*-

##
# Copyright (с) Ildar Bikmamatov 2022
# License: MIT
# Source: https://github.com/bayrell-tutorials/perceptron-xor
# Link:
# https://blog.bayrell.org/ru/iskusstvennyj-intellekt/411-obuchenie-mnogoslojnogo-perseptrona-operaczii-xor.html
##

import torch
from torch import nn
from torchsummary import summary


# Detect device for tensor
tensor_device = torch.device('cpu') 
#tensor_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  


# Step 1. Prepare DataSet
data_train = [
	{ "in": [0, 0], "out": [0] },
	{ "in": [0, 1], "out": [1] },
	{ "in": [1, 0], "out": [1] },
	{ "in": [1, 1], "out": [0] },
]


# Convert to question and answer DataSet
tensor_train_x = torch.tensor( list(map(lambda item: item["in"], data_train)) )
tensor_train_y = torch.tensor( list(map(lambda item: item["out"], data_train)) )

tensor_train_x = tensor_train_x.to(torch.float32).to(tensor_device)
tensor_train_y = tensor_train_y.to(torch.float32).to(tensor_device)

#print (tensor_train_x.shape)
#print (tensor_train_y.shape)

input_shape = 2
output_shape = 1


# Step 2. Create model
model = nn.Sequential(
	nn.Linear(input_shape, 16),
	nn.ReLU(),
	nn.Linear(16, output_shape),
	#nn.Softmax()
)

summary(model, (input_shape,))


# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.99))

# mean squared error
loss = nn.MSELoss()

# Batch size
batch_size = 2

# Epochs
epochs = 1000


# Step 3. Train model

print ("")
print ("")
print ("Start train")

history = []

model = model.to(tensor_device)

for i in range(epochs):
	
	# Model result
	model_res = model(tensor_train_x)
	
	# Loss value
	loss_value = loss(model_res, tensor_train_y)
	
	# Append to history
	loss_value_item = loss_value.item()
	history.append(loss_value_item)
	
	# Calc gradient
	optimizer.zero_grad()
	loss_value.backward()
	
	# Optimize
	optimizer.step()
	
	# Break if trained
	if loss_value_item < 0.01:
		break
	
	# Debug output
	if i % 10 == 0:
		print (f"{i+1},\t loss: {loss_value_item}")
	
	
# Show history

import matplotlib.pyplot as plt

plt.plot(history)
plt.title('Loss')
plt.show()


# Step 4. Control nn

print ("")
print ("")
print ("Control")

control_x = [
	[0, 0],
	[0, 1],
	[1, 0],
	[1, 1],
]

control_x = torch.tensor(control_x).to(torch.float32).to(tensor_device)

print ("Shape:", control_x.shape)

answer = model( control_x )

for i in range(len(answer)):
	print(control_x[i].tolist(), "->", answer[i].round().tolist())
