import argparse

def _setup_parser():
	parser = argparse.ArgumentParser(description="Convolutional Autoencoder (CAE) for MNIST dataset clustering")
	
	parser.add_argument('-s', '--scalers', nargs='+', type=str, help="Scalers to apply")

	return parser.parse_args()
