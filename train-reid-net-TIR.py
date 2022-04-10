from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import random
import copy
from black import delimiter_split
import numpy as np

from torchreid.data import ImageDataset
import torchreid

class NewDataset(ImageDataset):
    dataset_dir = 'new_dataset'

    def __init__(self, root='', **kwargs):
        self.root = osp.abspath(osp.expanduser(root))
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        # All you need to do here is to generate three lists,
        # which are train, query and gallery.
        # Each list contains tuples of (img_path, pid, camid),
        # where
        # - img_path (str): absolute path to an image.
        # - pid (int): person ID, e.g. 0, 1.
        # - camid (int): camera ID, e.g. 0, 1.
        # Note that
        # - pid and camid should be 0-based.
        # - query and gallery should share the same pid scope (e.g.
        #   pid=0 in query refers to the same person as pid=0 in gallery).
        # - train, query and gallery share the same camid scope (e.g.
        #   camid=0 in train refers to the same camera as camid=0
        #   in query/gallery).
        
        # Only using thermal cam
        cams = os.listdir("D:/thesis-data/SYSU-MM01")
        train = []
        query = []
        gallery = []
        all_pids = []

        # Extract all unique ids in dataset
        for cam in cams:
            if cam == 'cam3' or cam == 'cam6':
                camid = 0 if cam == 'cam3' else 1
                persons = os.listdir("D:/thesis-data/SYSU-MM01/" + cam)
                for person in persons:
                    images = os.listdir("D:/thesis-data/SYSU-MM01/" + cam + "/" + person)
                    pid = int(person)
                    all_pids.append(pid)
                    
        unique_pids = list(set(all_pids))
        unique_pids.sort()

        # Split train and test set based on only unique pids, thus an id cannot be in both train and test set
        random.shuffle(unique_pids)
        train_pids = unique_pids[:int(len(unique_pids) * 0.8)]
        test_pids = unique_pids[int(len(unique_pids) * 0.8):]

        # Reset mapping
        train_hash =  {k : i for i,k in enumerate(train_pids)}
        test_hash = {k : i for i,k in enumerate(test_pids)}

        # Add image path, id, and camid to train, query and gallery set
        for cam in cams:
            if cam == 'cam3' or cam == 'cam6':
                camid = 0 if cam == 'cam3' else 1
                persons = os.listdir("D:/thesis-data/SYSU-MM01/" + cam)
                for person in persons:
                    images = os.listdir("D:/thesis-data/SYSU-MM01/" + cam + "/" + person)
                    pid = int(person)                    
                    for image in images:
                        if pid in train_pids:
                            train.append((
                                "D:/thesis-data/SYSU-MM01/" + cam + "/" + person + "/" + image,
                                train_hash[pid],
                                camid 
                            ))
                        else:
                            if camid == 0:
                                query.append((
                                    "D:/thesis-data/SYSU-MM01/" + cam + "/" + person + "/" + image,
                                    test_hash[pid],
                                    camid 
                                ))  
                            else:
                                gallery.append((
                                "D:/thesis-data/SYSU-MM01/" + cam + "/" + person + "/" + image,
                                test_hash[pid],
                                camid 
                            ))


        super(NewDataset, self).__init__(train, query, gallery, **kwargs)

if __name__ == '__main__':
    torchreid.data.register_image_dataset('SYSU-MM01', NewDataset)

    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='SYSU-MM01',
        targets='SYSU-MM01',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=32,
        combineall=False,
        transforms=['random_flip', 'random_crop']
    )
    
    
    model = torchreid.models.build_model(
        name='resnet50',
        num_classes=datamanager.num_train_pids,
        loss='softmax',
        pretrained=True
    )

    model = model.cuda()

    optimizer = torchreid.optim.build_optimizer(
        model,
        optim='adam',
        lr=0.0003
    )

    scheduler = torchreid.optim.build_lr_scheduler(
        optimizer,
        lr_scheduler='single_step',
        stepsize=20
    )

    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )

    engine.run(
        save_dir='log/resnet50',
        max_epoch=60,
        start_eval = 1,
        eval_freq=1,
        print_freq=10,
        test_only=False,
        )
    