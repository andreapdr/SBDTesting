import pandas as pd
from argparse import ArgumentParser

def main(args):
	data = pd.read_csv(args.input)
	print("Data loaded successfully:")
	print(data.head())

	exit(0)

if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("--input", type=str, required=False, default="sample.csv", help="Path to the input CSV file")
	args = parser.parse_args()
	main(args)
