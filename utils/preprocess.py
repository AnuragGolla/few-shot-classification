import numpy as np
import os
from collections import defaultdict
from glob import glob
from skimage import io, transform
import tqdm
import torch

# Global parameters


# Data loading parameters only


def preprocess(dataset):
    """
    args:
    - dataset: string representing dataset to be used (omniglot or miniimagenet)
    """
    if dataset == 'omniglot':
        H, W = 28, 28    # Images will be resized to this height and width

        # Data loading parameters
        num_writers = 20    # Number of drawings per character
        num_alphabets = 50   # Number of alphabets in Omniglot (explained below)
        num_characters = 1623 # Number of characters in the entire Omniglot

    root = '.'
    data_path = os.path.join(root, os.path.join('../data', dataset))
    cache_path = os.path.join(root, os.path.join('../data/cache', dataset))
    os.makedirs(cache_path, exist_ok=True)
    dataset_path = os.path.join(cache_path, 'dataset.npy')

    if os.path.exists(dataset_path): # We've consolidated the data before, just load it in
        alphabets = np.load(dataset_path, allow_pickle=True).item()
    else:
        image_paths = glob(os.path.join(data_path, 'images_background/*/*/*.png'))
        image_paths.extend(glob(os.path.join(data_path, 'images_evaluation/*/*/*.png')))

        image_paths = sorted(image_paths)
        alphabets = defaultdict(list)
        for image_path in image_paths:
            *_, alphabet, character, drawing = image_path.split('/')
            # Here we load in images and resize all images to H by W with skimage
            alphabets[alphabet].append(transform.resize(io.imread(image_path), [H, W]))

        # Omniglot, with 1623 total classes, is actually divided into 50 variable-sized alphabets
        alphabets = {alphabet: np.stack(images).reshape(-1, num_writers, H, W) for alphabet, images in alphabets.items()}
        np.save(dataset_path, alphabets)

    # Normally it's good to center and normalize data, but here since each image's white background
    # has greyscale value 1.0 and the black writing has 0.0, we simply do 1 - the image to make most of the image
    # 0.0 and the writing 1.0
    processed_alphabets = {n: 1 - alphabet for n, alphabet in alphabets.items()}

    # Here we split 25% alphabets into the validation and test meta-datasets, each.
    # This is actually not what most of the papers do; they usually directly split the 1623 characters instead
    # of alphabets
    rng = np.random.RandomState(0)
    alphabet_names = list(alphabets)
    rng.shuffle(alphabet_names) # randomize the alphabets 
    num_train, num_val = int(num_alphabets * 0.5), int(num_alphabets * 0.25)

    # Train, val, test sets
    train_val_test_splits = np.split(alphabet_names, [num_train, num_train + num_val])
    sets = [{n: processed_alphabets[n] for n in names} for names in train_val_test_splits]
    train_alphabets, val_alphabets, test_alphabets = sets
    print('Number of alphabets in train, validation, test:', [len(x) for x in sets])
    print('Number of characters in train, validation, test:', [sum(len(v) for v in x.values()) for x in sets])

    return train_alphabets, val_alphabets, test_alphabets

