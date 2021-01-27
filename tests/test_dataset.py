import unittest
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import YinYangDataset
from torch.utils.data import DataLoader


class TestDataset(unittest.TestCase):
    def test_setup(self):
        # check that basic init does not crash
        train_set = YinYangDataset(size=5000, seed=42)

    def test_getitem(self):
        # check __getitem__
        elem_idx = 17
        train_set = YinYangDataset(size=5000, seed=42)
        vals, label = train_set[elem_idx]

        self.assertListEqual(list(vals), list(train_set._YinYangDataset__vals[elem_idx]))
        self.assertEqual(label, train_set._YinYangDataset__cs[elem_idx])

    def test_sample_structure(self):
        # check that vals have 4 values
        train_set = YinYangDataset(size=5000, seed=42)
        for sample in train_set:
            vals, label = sample
            self.assertEqual(len(vals), 4)
            # check that label is an integer between 0 and 2
            self.assertIsInstance(label, int)
            self.assertTrue(0 <= label <= 2)

    def test_len(self):
        # check __len__
        train_set = YinYangDataset(size=5000, seed=42)
        len_1 = len(train_set)
        len_2 = 0
        for _, _ in enumerate(train_set):
            len_2 += 1
        self.assertEqual(len_1, len_2)

    def test_seeding(self):
        # check that different seeds produce different datasets
        set1 = YinYangDataset(size=1000, seed=42)
        set2 = YinYangDataset(size=1000, seed=41)
        for i, elem in enumerate(set1):
            vals1, label1 = elem
            vals2, label2 = set2[i]
            for j in range(4):
                self.assertNotEqual(vals1[j], vals2[j])
            # only check if input values are different, labels could be the same by chance
        # check that same seeds produce same datasets
        set3 = YinYangDataset(size=1000, seed=40)
        set4 = YinYangDataset(size=1000, seed=40)
        for i, elem in enumerate(set3):
            vals3, label3 = elem
            vals4, label4 = set4[i]
            self.assertListEqual(list(vals3), list(vals4))
            self.assertEqual(label3, label4)

    def test_torchloader(self):
        # check that it can be used for torch dataloaders
        train_set = YinYangDataset(size=5000, seed=42)
        train_loader = DataLoader(train_set, batch_size=20, shuffle=True)

    def test_plot(self):
        # generate plot with default params for visual check
        dataset_train = YinYangDataset(size=5000, seed=42)
        dataset_validation = YinYangDataset(size=1000, seed=41)
        dataset_test = YinYangDataset(size=1000, seed=40)
        batchsize_train = 20
        batchsize_eval = len(dataset_test)

        train_loader = DataLoader(dataset_train, batch_size=batchsize_train, shuffle=True)
        val_loader = DataLoader(dataset_validation, batch_size=batchsize_eval, shuffle=True)
        test_loader = DataLoader(dataset_test, batch_size=batchsize_eval, shuffle=False)

        fig, axes = plt.subplots(ncols=3, sharey=True, figsize=(15, 8))
        titles = ['Training set', 'Validation set', 'Test set']
        for i, loader in enumerate([train_loader, val_loader, test_loader]):
            axes[i].set_title(titles[i])
            axes[i].set_aspect('equal', adjustable='box')
            xs = []
            ys = []
            cs = []
            for batch, batch_labels in loader:
                for j, item in enumerate(batch):
                    x1, y1, x2, y2 = item
                    c = batch_labels[j]
                    xs.append(x1)
                    ys.append(y1)
                    cs.append(c)
            xs = np.array(xs)
            ys = np.array(ys)
            cs = np.array(cs)
            axes[i].scatter(xs[cs == 0], ys[cs == 0], color='C0', edgecolor='k', alpha=0.7, s=88)
            axes[i].scatter(xs[cs == 1], ys[cs == 1], color='C1', edgecolor='k', alpha=0.7, s=88)
            axes[i].scatter(xs[cs == 2], ys[cs == 2], color='C2', edgecolor='k', alpha=0.7, s=88)
            axes[i].set_xlabel('x1')
            if i == 0:
                axes[i].set_ylabel('y1')
            fig.tight_layout()
            plt.savefig('tests/testing_plot.png')

