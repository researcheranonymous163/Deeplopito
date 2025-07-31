from os import environ
from random import Random

import torch
import torch.utils.data
import torchvision.transforms as transforms
from dfl.data import stratified_split
from torchvision import datasets

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

_download = environ.get('DATA_DOWNLOAD', 'False').lower() in {'true', '1', 't', 'y', 'yes'}
train_data_all = datasets.MNIST(root='./data/', train=True, download=_download, transform=transform)
test_data_all = datasets.MNIST(root='./data', train=False, download=_download, transform=transform)


def stratified(index: int, n_splits: int = 10):
	train_splits = stratified_split(
		list(range(len(train_data_all))),
		labels=train_data_all.targets,
		n_splits=n_splits,
		rnd=Random(1234)
	)
	
	test_splits = stratified_split(
		list(range(len(test_data_all))),
		labels=test_data_all.targets,
		n_splits=n_splits,
		rnd=Random(1234)
	)
	
	return torch.utils.data.Subset(train_data_all, train_splits[index]), torch.utils.data.Subset(test_data_all, test_splits[index])


_train_class_map = {i: torch.nonzero(train_data_all.targets == i, as_tuple=False).squeeze().tolist() for i in range(len(train_data_all.classes))}
_test_class_map = {i: torch.nonzero(test_data_all.targets == i, as_tuple=False).squeeze().tolist() for i in range(len(test_data_all.classes))}


def one_ten(index: int):
	return torch.utils.data.Subset(train_data_all, _train_class_map[index]), torch.utils.data.Subset(test_data_all, _test_class_map[index])


if __name__ == '__main__':
	# Just downloads the data if the `DATA_DOWNLOAD` environment variable is set.
	pass
