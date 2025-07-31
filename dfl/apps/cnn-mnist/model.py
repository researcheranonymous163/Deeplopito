import torch.nn as nn


def mnist_2nn():
	return nn.Sequential(
		nn.Flatten(),  # Flatten the input (28x28 -> 784)
		nn.Linear(784, 200),  # First hidden layer
		nn.ReLU(),  # ReLU activation
		nn.Linear(200, 200),  # Second hidden layer
		nn.ReLU(),  # ReLU activation
		nn.Linear(200, 10)  # Output layer (logits for 10 classes)
	)


def mnist_cnn():
	return nn.Sequential(
		nn.Conv2d(1, 32, kernel_size=5),  # First conv layer: 1 input channel, 32 output channels, 5x5 kernel
		nn.ReLU(),  # ReLU activation
		nn.MaxPool2d(kernel_size=2),  # 2x2 Max Pooling
		nn.Conv2d(32, 64, kernel_size=5),  # Second conv layer: 32 input channels, 64 output channels, 5x5 kernel
		nn.ReLU(),  # ReLU activation
		nn.MaxPool2d(kernel_size=2),  # 2x2 Max Pooling
		nn.Flatten(),  # Flatten the output of the conv layers
		nn.Linear(64 * 4 * 4, 512),  # Fully connected layer (1024 input features -> 512 units)
		nn.ReLU(),  # ReLU activation
		nn.Linear(512, 10)  # Output layer (logits for 10 classes)
	)
