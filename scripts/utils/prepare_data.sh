#!/bin/bash
#python make_hdf5.py --dataset I128 --batch_size 256 --data_root data
python calculate_inception_moments.py --dataset cgan --data_root '/content/drive/MyDrive' --num_workers 4
