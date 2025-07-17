"""
Print the contents of an NPZ file.
"""
from numpy import load

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Open and inspect an NPZ file.")
    parser.add_argument('npz_file', type=str, help='Path to the NPZ file to open')

    args = parser.parse_args()

    data = load(args.npz_file, allow_pickle=True)
    lst = data.files
    for item in lst:
        print(item)
        print(data[item])