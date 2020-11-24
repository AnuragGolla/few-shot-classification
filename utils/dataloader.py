import numpy as np


def from_torch(x):
    """
    Convert from a torch tensor to numpy array
    """
    return x.detach().cpu().numpy()

class OmniglotLoader:
    def __init__(self, alphabets, batch_size, n_way=5, k_shot=1, q_queryperclass=1, augment_rotate=False, augment_flip=False):
        """
        Since most few-shot learning papers just mix the alphabets together when sampling characters for tasks,
        you should just concatenate the characters in different alphabets together here

        alphabets: a dictionary mapping from alphabet names to numpy arrays of size
            (num_characters, num_writers, H, W), where num_characters is the number of characters in the alphabet
        batch_size: number of tasks to generate at once
        augment_rotate: whether to augment the data by rotating each alphabet 0, 90, 180, and 270 degrees
        augment_flip: whether to augment the data by flipping each alphabet horizontally
        """
        self.data = {}
        self.batch_size = batch_size
        self.rotate_angs = [0,90, 180, 270]
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_queryperclass = q_queryperclass
        for alphabet in alphabets:
            ori= alphabets[alphabet].shape
            alphabet_temp = alphabets[alphabet]
            if augment_flip:
                flipped = np.flip(alphabet_temp, axis=-1)
                alphabet_temp = np.concatenate((alphabet_temp, flipped))
            if augment_rotate:
                rotate_90 = np.rot90(alphabet_temp, axes=(-2, -1))
                rotate_180 = np.rot90(alphabet_temp, axes=(-2, -1), k=2)
                rotate_270 = np.rot90(alphabet_temp, axes=(-2, -1), k=3)
                alphabet_temp = np.concatenate((alphabet_temp, rotate_90, rotate_180, rotate_270))

            self.data[alphabet] = alphabet_temp

    def __iter__(self):
        """
        Define this class as an iterable that yields a batch of tasks at each iteration when used in a for-loop.
        Each task in the batch should have N_way classes (characters) randomly sampled from the meta-dataset.
        Each class should have K_shot + Q_queryperclass images; the first K_shot images are the support set 
        and the last Q_queryperclass images are the queries

        return: a batch of tasks, which should be a numpy array shaped
            (batch_size, N_way, K_shot + Q_queryperclass, H, W)
        """
        num_class = len(self.data)
        while True:
            batch = []
            for _ in range(self.batch_size):
                random_alphabet_name = np.random.choice(list(self.data.keys()), size = self.n_way)

                random_chars = [self.data[alphabet][np.random.choice(len(self.data[alphabet]), size=1, replace=False),:,:,:] for alphabet in random_alphabet_name]
                random_chars = np.concatenate(random_chars, axis=0)

                key_query_index = np.random.choice(random_chars.shape[1], size = self.k_shot + self.q_queryperclass, replace=False)

                supports_queries = random_chars[:,key_query_index,:,:]

                batch.append(supports_queries)

            yield np.array(batch) # Shape (batch_size, N_way, K_shot + Q_queryperclass, H, W)

class MiniImageNetLoader:
    def __init__(self, task_set, batch_size, k_shot, n_way):
        self.task_set = task_set
        self.batch_size = batch_size
        self.k_shot = k_shot
        self.n_way = n_way

        self.channels = 3
        self.H = 84
        self.W = 84

    def __iter__(self):
        while True:
            batch = []
            for _ in range(self.batch_size):
                X, _ = self.task_set.sample()
                queries = from_torch(X.view(self.n_way, self.k_shot, self.channels, self.H, self.W))
                batch.append(queries)

            yield np.array(batch)