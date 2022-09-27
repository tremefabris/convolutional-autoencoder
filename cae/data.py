from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import DataLoader, random_split

class MNISTDataCarrier():
	def __init__(self, split=(.8, .1, .1), dataset_dir='datasets/', transformations=[ToTensor()]):
		assert sum(split) == 1, "Splits must sum up to one"

		#self.train_split, self.val_split, self.test_split = split
		self.split = split
		self.dataset_dir = dataset_dir
		self.transformations = transformations
		self.built = False

	def _load_data(self):
		T = Compose(self.transformations)
		self.dataset = MNIST(self.dataset_dir, train=True, download=True, transform=T)

	def _split_data(self):
		sizes = np.array(self.split) * len(dataset)
		train_size = int( sizes[0] )
		val_size   = int( sizes[1] )
		test_size  = int( sizes[2] )
		
		tr, v, t = random_split(self.dataset, [train_size, val_size, test_size])
		self.trainset = tr
		self.valset   = v
		self.testset  = t

	def build(self):
		self._load_data()
		self._split_data()
		self.built = True

	def retrieve_data(self, batch_size=32)
		if not self.built:
			self.build()
		TR = DataLoader(self.trainset, batch_size=batch_size, shuffle=True)
		V  = DataLoader(self.valset, batch_size=batch_size, shuffle=False)
		T  = DataLoader(self.testset, batch_size=batch_size, shuffle=False)


