# Copyright (c) OpenMMLab. All rights reserved.
from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.builder import build_dataset

import pdb

@DATASETS.register_module()
class SemiDataset:
    def __init__(self, sup_dataset, unsup_dataset, default_args=None):
        self.sup_dataset = build_dataset(sup_dataset, default_args)
        self.unsup_dataset = build_dataset(unsup_dataset, default_args)
        tmp_sup_dataset = self.sup_dataset.datasets[0] if self.sup_dataset.__class__.__name__ == "ConcatDataset" else self.sup_dataset
        tmp_unsup_dataset = self.unsup_dataset.datasets[0] if self.unsup_dataset.__class__.__name__ == "ConcatDataset" else self.unsup_dataset
        self.ignore_index = tmp_sup_dataset.ignore_index
        self.CLASSES = tmp_sup_dataset.CLASSES
        self.PALETTE = tmp_sup_dataset.PALETTE
        assert tmp_sup_dataset.ignore_index == tmp_unsup_dataset.ignore_index
        assert tmp_sup_dataset.CLASSES == tmp_unsup_dataset.CLASSES
        assert tmp_sup_dataset.PALETTE == tmp_unsup_dataset.PALETTE

    def __len__(self):
        return len(self.sup_dataset) * len(self.unsup_dataset)
    def __getitem__(self, idx):
        sup_data = self.sup_dataset[idx // len(self.unsup_dataset)]
        unsup_data = self.unsup_dataset[idx % len(self.unsup_dataset)]
        return {"sup_data":sup_data, "unsup_data": unsup_data}

