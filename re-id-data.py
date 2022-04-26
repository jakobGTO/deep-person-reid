from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import os
import os.path as osp
import random
import copy
import numpy as np

from torchreid.data import ImageDataset
import torchreid

class RGBDataset(ImageDataset):
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
      
        train = []
        query = []
        gallery = []

        cams = [i for i in os.listdir("D:/thesis-data/SYSU-MM01/RGB/") if i.startswith("cam")]
        
        train_idx = np.loadtxt("D:/thesis-data/SYSU-MM01/train_id.txt", delimiter=",", dtype = int)
        test_idx = np.loadtxt("D:/thesis-data/SYSU-MM01/test_id.txt", delimiter=",", dtype = int)
        val_idx = np.loadtxt("D:/thesis-data/SYSU-MM01/val_id.txt", delimiter=",", dtype = int)
        all_idx = np.loadtxt("D:/thesis-data/SYSU-MM01/available_id.txt", delimiter=",", dtype = int)
            
        cam_table = {'cam1': 0, 'cam2': 1, 'cam4': 2, 'cam5': 3}
        
        pid_container_train = set()
        pid_container_test = set()
        
        for cam in cam_table:
            persons = os.listdir("D:/thesis-data/SYSU-MM01/RGB/" + cam)        
            for person in persons:
                pid = int(person)
                if pid in train_idx:
                    pid_container_train.add(pid)
                else:
                    pid_container_test.add(pid)
        
        pid2label_train = {pid: label for label, pid in enumerate(pid_container_train)}
        pid2label_test = {pid: label for label, pid in enumerate(pid_container_test)}
        
        for cam in cam_table:
            camid = cam_table[cam]
            persons = os.listdir("D:/thesis-data/SYSU-MM01/RGB/" + cam)
            for person in persons:
                pid = int(person)
                images = os.listdir("D:/thesis-data/SYSU-MM01/RGB/" + cam + "/" + person)
                for image in images:
                    if pid in train_idx:
                        train.append((
                            "D:/thesis-data/SYSU-MM01/RGB/" + cam + "/" + person + "/" + image,
                            pid2label_train[pid],
                            camid 
                        ))
                    else:
                        if camid == 0:
                            query.append((
                                "D:/thesis-data/SYSU-MM01/RGB/" + cam + "/" + person + "/" + image,
                                pid2label_test[pid],
                                camid
                            ))
                        elif camid == 1:
                            gallery.append((
                                "D:/thesis-data/SYSU-MM01/RGB/" + cam + "/" + person + "/" + image,
                                pid2label_test[pid],
                                camid 
                            ))

        super(RGBDataset, self).__init__(train, query, gallery, **kwargs)

class TIRDataset(ImageDataset):
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
      
        train = []
        query = []
        gallery = []

        cams = [i for i in os.listdir("D:/thesis-data/SYSU-MM01/TIR/") if i.startswith("cam")]
        
        train_idx = np.loadtxt("D:/thesis-data/SYSU-MM01/train_id.txt", delimiter=",", dtype = int)
        test_idx = np.loadtxt("D:/thesis-data/SYSU-MM01/test_id.txt", delimiter=",", dtype = int)
        val_idx = np.loadtxt("D:/thesis-data/SYSU-MM01/val_id.txt", delimiter=",", dtype = int)
        all_idx = np.loadtxt("D:/thesis-data/SYSU-MM01/available_id.txt", delimiter=",", dtype = int)
            
        cam_table = {'cam3': 0, 'cam6': 1}
        
        pid_container_train = set()
        pid_container_test = set()
        
        for cam in cam_table:
            persons = os.listdir("D:/thesis-data/SYSU-MM01/TIR/" + cam)        
            for person in persons:
                pid = int(person)
                if pid in train_idx:
                    pid_container_train.add(pid)
                else:
                    pid_container_test.add(pid)
        
        pid2label_train = {pid: label for label, pid in enumerate(pid_container_train)}
        pid2label_test = {pid: label for label, pid in enumerate(pid_container_test)}
        
        for cam in cam_table:
            camid = cam_table[cam]
            persons = os.listdir("D:/thesis-data/SYSU-MM01/TIR/" + cam)
            for person in persons:
                pid = int(person)
                images = os.listdir("D:/thesis-data/SYSU-MM01/TIR/" + cam + "/" + person)
                for image in images:
                    if pid in train_idx:
                        train.append((
                            "D:/thesis-data/SYSU-MM01/TIR/" + cam + "/" + person + "/" + image,
                            pid2label_train[pid],
                            camid 
                        ))
                    else:
                        if camid == 1:
                            query.append((
                                "D:/thesis-data/SYSU-MM01/TIR/" + cam + "/" + person + "/" + image,
                                pid2label_test[pid],
                                camid
                            ))
                        else:
                            gallery.append((
                                "D:/thesis-data/SYSU-MM01/TIR/" + cam + "/" + person + "/" + image,
                                pid2label_test[pid],
                                camid 
                            ))

        super(TIRDataset, self).__init__(train, query, gallery, **kwargs)

def train_rgb_net():
    torchreid.data.register_image_dataset('SYSU-MM01', RGBDataset)
    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='SYSU-MM01',
        targets='SYSU-MM01',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop'],
        workers=1,
        combineall = False
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
        save_dir='Trained-RGB-model',
        max_epoch=100,
        eval_freq=10,
        print_freq=10,
        test_only=False
        )
    
def train_tir_net():
    torchreid.data.register_image_dataset('SYSU-MM01', TIRDataset)
    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='SYSU-MM01',
        targets='SYSU-MM01',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop'],
        workers=1,
        combineall = False
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
        save_dir='Trained-TIR-model',
        max_epoch=100,
        eval_freq=10,
        print_freq=10,
        test_only=False
        )
    
if __name__ == '__main__':
    #train_rgb_net()
    #train_tir_net()
    
    torchreid.data.register_image_dataset('SYSU-MM01', RGBDataset)
    
    datamanager = torchreid.data.ImageDataManager(
        root='reid-data',
        sources='SYSU-MM01',
        targets='SYSU-MM01',
        height=256,
        width=128,
        batch_size_train=32,
        batch_size_test=100,
        transforms=['random_flip', 'random_crop'],
        workers=1,
        combineall = False
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


    torchreid.utils.load_pretrained_weights(model, 'Trained-RGB-model/model/model.pth.tar-100')
    
    
    engine = torchreid.engine.ImageSoftmaxEngine(
        datamanager,
        model,
        optimizer=optimizer,
        scheduler=scheduler,
        label_smooth=True
    )


    
    engine.run(
        max_epoch=100,
        eval_freq=10,
        print_freq=10,
        test_only=True
        )