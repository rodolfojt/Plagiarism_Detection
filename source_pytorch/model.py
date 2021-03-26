# torch imports
import torch.nn as nn
import torch


## TODO: Complete this classifier
class BinaryClassifier(nn.Module):
	"""
	Define a neural network that performs binary classification.
	The network should accept your number of features as input, and produce 
	a single sigmoid value, that can be rounded to a label: 0 or 1, as output.
	
	Notes on training:
	To train a binary classifier in PyTorch, use BCELoss.
	BCELoss is binary cross entropy loss, documentation: https://pytorch.org/docs/stable/nn.html#torch.nn.BCELoss
	"""

	## TODO: Define the init function, the input params are required (for loading code in train.py to work)
	def __init__(self, input_features, hidden_dim, output_dim):
		"""
		Initialize the model by setting up linear layers.
		Use the input parameters to help define the layers of your model.
		:param input_features: the number of input features in your training/test data
		:param hidden_dim: helps define the number of nodes in the hidden layer(s)
		:param output_dim: the number of outputs you want to produce
		"""
		super(BinaryClassifier, self).__init__()

		# define any initial layers, here
		self.input_features = input_features
		self.hidden_dim = hidden_dim
		self.output_dim = output_dim
		
		self.fc1 = nn.Linear(self.input_features, self.hidden_dim)
		self.hidden1 = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.hidden2 = nn.Linear(self.hidden_dim, self.hidden_dim)
		self.outpt = nn.Linear(self.hidden_dim,self.output_dim)

		nn.init.xavier_uniform_(self.fc1.weight)
		nn.init.zeros_(self.fc1.bias)
		nn.init.xavier_uniform_(self.hidden1.weight)
		nn.init.zeros_(self.hidden1.bias)
		nn.init.xavier_uniform_(self.hidden2.weight)
		nn.init.zeros_(self.hidden2.bias)
		nn.init.xavier_uniform_(self.outpt.weight)
		nn.init.zeros_(self.outpt.bias)
		self.sig = nn.Sigmoid()

	
	## TODO: Define the feedforward behavior of the network
	def forward(self, x):
		"""
		Perform a forward pass of our model on input features, x.
		:param x: A batch of input features of size (batch_size, input_features)
		:return: A single, sigmoid-activated value as output
		"""
		
		# define the feedforward behavior
		x = x.t()
		lengths = x[0,:]
		x = x[1:,:]
		x = torch.tanh(self.fc1(x))
		x = torch.tanh(self.hidden1(x))
		x = torch.tanh(self.hidden2(x))	
		x = self.sig(self.outpt(x))
		return x.squeeze()
	