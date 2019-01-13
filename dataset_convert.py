import tensorflow as tf
import argparse
import sys
import os

DATASET_DIR = 'dataset'

parser = argparse.ArgumentParser(description='Convert CelebA data into TFRecordDataset file.')
parser.add_argument('folder', help="Directory that contains the CelebA images")
args = parser.parse_args()

# Read all file names
print(f'Listing files in {args.folder}...', end='')
file_list = os.listdir(args.folder)
total = len(file_list)
print(f'Found {total} files.')

# Prepare TFRecordDataset file
if not os.path.exists(DATASET_DIR):
    os.mkdir(DATASET_DIR) 
    print(f'Dataset directory has been created.')

max_length = 0
def log(filename, counter):
    global max_length
    message = f'{counter}/{total} Converting {filename}'
    max_length = max(len(message), max_length)
    print(f'\r{message.ljust(max_length - len(message))}', end='')

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

with tf.python_io.TFRecordWriter(os.path.join(DATASET_DIR, 'dataset.tr')) as writer:
    # Read and write images into dataset
    counter = 0
    for filename in file_list:
        counter += 1
        log(filename, counter)
        with open(os.path.join(args.folder, filename), 'rb') as fd:
            data = fd.read()
            example = tf.train.Example(features=tf.train.Features(feature={'img': _bytes_feature(data)}))
            writer.write(example.SerializeToString())

print(f'\nDone! {counter} files have been writen into TFRecord file.')