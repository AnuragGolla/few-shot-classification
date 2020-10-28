import torch
import numpy as np
import argparse
import os
import sys
from ../models/simple-cnaps import SimpleCnaps
from ../utils/dataloader import MetaDataLoader
from ../utils/metrics import Loss, Accuracy
import tensorflow as tf

# Run Config
N_TRAIN_T = 110000
N_VAL_T = 200
N_TEST_T = 600
VAL_FREQ = 10000

# Learner Class Definition
class Learner:

    def __init__(self):

        # Command Line Args
        self.args = self.parse_command_line()

        # File paths
        self.ckpt_dir, self.logfile, self.ckpt_valpath, self.ckpt_final = build_log_files(self.args.ckpt_dir)

        # Model and Data
        self.device = "cpu"
        self.model = self.build_model()
        self.trainD, self.valD, self.testD = self.init_data()
        self.metaD = MetaDataLoader(self.args.data_path, self.args.mode, self.trainD, self.valD,
                                    self.testD)

        # Learn Params
        self.loss_fn = Loss
        self.accuracy_fn = Accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()

    def build_model(self):
        model = SimpleCnaps(device=self.device, args=self.args).to(self.device)
        model.train()
        model.feature_extractor.eval()
        return model

    def init_data(self):
        trainD = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
        valD = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                          'mscoco']
        testD = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
                    'traffic_sign', 'mscoco', 'mnist', 'cifar10', 'cifar100']
        return trainD, valD, testD

    def parse_command_line(self):
        parser = argparse.ArgumentParser()

        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--pretrained_resnet_path", default="../models/pretrained_resnet.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--ckpt_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")

        args = parser.parse_args()
        return args

    def build_log_files(ckpt_dir):
        ckpt_dir = os.path.join(ckpt_dir, datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        if not os.path.exists(ckptt_dir):
            os.makedirs(ckpt_dir)
        ckpt_path_val = os.path.join(ckpt_dir, 'best_val.pt')
        ckpt_path_final = os.path.join(ckpt_dir, 'final.pt')
        logfp = os.path.join(ckpt_dir, 'log')
        logf = open(logfp, "w")
        return ckpt_dir, logf, ckpt_path_val, ckpt_path_final

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            if self.args.mode == 'train' or self.args.mode == 'train_test':
                train_accs = []
                losses = []
                total_itrs = N_TRAIN_T
                for it in range(total_itrs):
                    torch.set_grad_enabled(True)
                    task_dict = self.metaD.g.get_train_task(session)
                    task_loss, task_acc = self.train_task(task_dict)
                    train_accs.append(task_acc)
                    losses.append(task_loss)

                    if ((it + 1) % self.args.tasks_per_batch == 0) or (it == (total_iterations - 1)):
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    if (iteration + 1) % 1000 == 0:
                        # TODO: add log functionality
                        train_accs = []
                        losses = []

                    if ((it + 1) % VAL_FREQ == 0) and (it + 1) != total_itrs:
                        accuracy_dict = self.validate(session)
                        # TODO: add log functionality

                torch.save(self.model.state_dict(), self.ckpt_final)

            if self.args.mode == 'train_test':
                self.test(self.ckpt_final, session)
                self.test(self.ckpt_valpath, session)

            if self.args.mode == 'test':
                self.test(self.args.test_model_path, session)

            self.logfile.close()

    def train_task(self, task_dict):
        support_x, query_x, support_y, query_y = self.prepare_task(task_dict)
        query_logits = self.model(support_x, support_y, query_x)
        task_loss = self.loss_fn(query_logits, query_y, self.device) / self.args.tasks_per_batch
        regularization = (self.model.adaptation_network.regularization())
        regularizer_k = 0.001
        task_loss += regularizer_k * regularization
        task_acc = self.acc_fn(query_logits, query_y)
        task_loss.backward(retain_graph=False)
        return task_loss, task_acc

    def validate(self, session):
        with torch.no_grad():
            acc_dict ={}
            for item in self.valD:
                accs = []
                for _ in range(N_VAL_T):
                    task_dict = self.metaD.g.get_validation_task(item, session)
                    support_x, query_x, support_y, query_y = self.prepare_task(task_dict)
                    query_logits = self.model(support_x, support_y, query_x)
                    acc = self.acc_fn(query_logits, query_y)
                    acc.append(acc.item())
                    del query_logits
                acc = np.array(accs).mean() * 100.0
                conf = (196.0 * np.array(accs).std()) / np.sqrt(len(accs))
                acc_dict[item] = {"acc": acc, "conf": conf}
        return acc_dict

    def test(self, path, session):
        self.model = self.build_model()
        self.model.load_state_dict(torch.load(path))
        with torch.no_grad():
            for item in self.testD:
                accs = []
                for _ in range(N_TEST_T):
                    task_dict = self.metaD.g.get_test_task(item, session)
                    support_x, query_x, support_y, query_y = self.prepare_task(task_dict)
                    query_logits = self.model(support_x, support_y, query_x)
                    acc = self.accuracy_fn(query_logits, query_y)
                    accs.append(acc.item())
                    del query_logits
                acc = np.array(accs).mean() * 100.0
                conf = (196.0 * np.array(accs).std()) / np.sqrt(len(accs))
                # TODO: add log functionality

    def prepare_task(self, task_dict):
        support_x_np, support_y_np = task_dict['support_x'], task_dict['support_y']
        query_x_np, query_y_np = task_dict['query_x'], task_dict['query_y']

        support_x_np = support_x_np.transpose([0, 3, 1, 2])
        support_x_np, support_y_np = self.shuffle(support_x_np, support_y_np)
        support_x = torch.from_numpy(support_x_np)
        support_y = torch.from_numpy(support_y_np)

        query_x_np = query_x_np.transpose([0, 3, 1, 2])
        query_x_np, query_y_np = self.shuffle(query_x_np, query_y_np)
        query_x = torch.from_numpy(query_x_np)
        query_y = torch.from_numpy(query_y_np)

        support_x = support_x.to(self.device)
        query_x = query_x.to(self.device)
        support_y = support_y.to(self.device)
        query_y = query_y.type(torch.LongTensor).to(self.device)

        return support_x, query_x, support_y, query_y

    def shuffle(self, images, labels):
        perm = np.random.permutation(images.shape[0])
        return images[perm], labels[perm]

if __name__ == "__main__":
    learner = Learner()
    learner.run()
