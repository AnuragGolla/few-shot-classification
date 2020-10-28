from preprocess import preprocess
from data_loader import OmniglotLoader
from network import Network
import pdb
import os
import torch
import tqdm
from distance import euclidean_distance, mahalanobis_distance

K_SHOT = 5
device = 'cuda'

def from_torch(x):
    """
    Convert from a torch tensor to numpy array
    """
    return x.detach().cpu().numpy()

def train(net, train_metadataset, num_steps):
    """
    Train the input neural network for num_steps
    
    net: an instance of MatchingNetwork, PrototypicalNetwork, or RelationNetwork
    num_steps: number of batches to train for
    
    return: the trained net, the training accuracies per step, and the validation accuracies per 100 steps
    """
    net = net.to(device)
    

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    train_accuracies = []
    val_accuracies = []
    for step, tasks in tqdm.tqdm(zip(range(num_steps), train_metadataset)):
        # Here you need to define the labels for a batch of tasks. Remember from exercise1.pdf that for each task,
        # we're remapping the original classes to classes 1, ..., N.
        # Since Python uses 0-indexing, you should remap the classes to 0, ..., N - 1
        # After defining the labels, call the forward pass of net on the tasks and labels
        ### Your code here ###
        labels = torch.LongTensor([[i] for i in range(5)]).view(-1,5).repeat(tasks.shape[0],1).to(device)
                        
        loss, accuracy = net.forward(tasks, labels)

        opt.zero_grad()
        loss.backward()
        opt.step()
        train_loss, train_accuracy = map(from_torch, (loss, accuracy))
        train_accuracies.append(train_accuracy)
        if (step + 1) % 100 == 0:
            val_loss, val_accuracy = evaluate(net, val_metadataset)
            val_accuracies.append(val_accuracy)
            print('step=%s   train(loss=%.5g, accuracy=%.5g)  val(loss=%.5g, accuracy=%.5g)' % (
                step + 1, train_loss, train_accuracy, val_loss, val_accuracy
            ))
    return net, train_accuracies, val_accuracies

def evaluate(net, metadataset, N_way=5):
    """
    Evalate the trained net on either the validation or test metadataset
    
    net: an instance of MatchingNetwork, PrototypicalNetwork, or RelationNetwork
    metadataset: validation or test metadataset
    
    return: a tuple (loss, accuracy) of Python scalars
    """
    with torch.no_grad(): # Evaluate without gradients
        tasks = next(iter(metadataset)) # Since our evaluation batch_size is so big, we only need one batch
        # Evaluate the net on the batch of tasks. You have to define the labels here too
        ### Your code here ###
        labels = torch.Tensor([[i] for i in range(N_way)]).view(-1,N_way).repeat(tasks.shape[0],1).to(device)
        loss, accuracy = net.forward(tasks, labels.long())

        loss, accuracy = map(from_torch, (loss, accuracy))
    return loss, accuracy

if __name__ == '__main__':
    train_alphabets, val_alphabets, test_alphabets = preprocess('omniglot')
    # We use a batch size of 64 while training
    train_metadataset = OmniglotLoader(train_alphabets, 32, k_shot=K_SHOT, augment_rotate=True, augment_flip=True)
    # We can use a batch size of 1600 for validation or testing
    val_metadataset = OmniglotLoader(val_alphabets, 32, k_shot=K_SHOT)
    test_metadataset = OmniglotLoader(test_alphabets, 32, k_shot=K_SHOT)

    train_steps = 500
    mn_net, mn_train_accuracies, mn_val_accuracies = train(Network(distance_function=mahalanobis_distance), train_metadataset, train_steps)