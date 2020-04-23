import matplotlib.pyplot as plt
import numpy as np
import cv2

from itertools import groupby
import os
import sys
import glob
import time

def get_mvs(prev_path, cur_path):
    ref_img = cv2.imread(prev_path, 0)
    cur_img = cv2.imread(cur_path, 0)
    flow = cv2.calcOpticalFlowFarneback(ref_img, cur_img, None, 0.5, 3, 8, 3, 16, 1.5, 0)
    return np.array([flow[..., 0], flow[..., 1]])

def calc_group(group, of_path):
    fn = group[0].split('/')[-1].split('.')[0][:-4] # filename
    
    # positions
    positions = np.array([[1, 7, 13],
                          [1, 4, 7], [7, 10, 13],
                          [1, 2, 4], [4, 3, 1], [4, 5, 7], [7, 6, 4],
                          [7, 8, 10], [10, 9, 7], [10, 11, 13], [13, 12, 10]])
    
    for g in range(len(group) // 12):
        for pos in positions:
            idx = pos - 1
            bmvs = get_mvs(group[12*g + idx[0]], group[12*g + idx[1]]) # before
            amvs = get_mvs(group[12*g + idx[2]], group[12*g + idx[1]]) # after

            # rearrange data
            # print(bmvs[0].astype(np.int8))
            bmvs /= 2
            # bmvs = bmvs.astype(np.int)
            bmvs += 128
            amvs /= 2
            # amvs = bmvs.astype(np.int)
            amvs += 128

            # save optical flows
            cv2.imwrite(of_path + fn + str(12*g + pos[1]).zfill(4) + '_before_flow_x_0001.jpg', bmvs[0])
            cv2.imwrite(of_path + fn + str(12*g + pos[1]).zfill(4) + '_before_flow_y_0001.jpg', bmvs[1])
            cv2.imwrite(of_path + fn + str(12*g + pos[1]).zfill(4) + '_after_flow_x_0001.jpg', amvs[0])
            cv2.imwrite(of_path + fn + str(12*g + pos[1]).zfill(4) + '_after_flow_y_0001.jpg', amvs[1])


if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: python get_opticalflow.py [list_path] [source_path] [target_dir] [start] [end]')
        exit()
    list_path = sys.argv[1]
    source_dir = sys.argv[2]
    target_dir = sys.argv[3]
    start_point = sys.argv[4]
    end_point = sys.argv[5]

    img_paths = []
    for path in glob.glob(source_dir + '*.png'):
        img_paths.append(path)

    # img_paths.sort(key=lambda p: (p.split('/')[-1].split('.')[0][:-4], int(p.split('/')[-1].split('.')[0][-4:]))) # sort by (name, frame)
    # img_paths = [list(i) for j, i in groupby(img_paths, lambda p: p.split('/')[-1].split('.')[0][:-4])] # group path by name

    # for g_idx in range(len(img_paths)):
    #     calc_group(img_paths[g_idx], target_dir)

    fn_list = np.genfromtxt(list_path, dtype=str)[int(start_point)-1:int(end_point)]

    for ind, fn in enumerate(fn_list):
        if source_dir + fn + '_0001.png' in img_paths:
            try:
                group = []
                for i in range(97):
                    group.append(source_dir + fn + '_%04d.png'%(i+1))
                calc_group(group, target_dir)
            except Exception as err:
                print('Handling run-time error:', err)
            else:
                print('[{}]({}) Finished.'.format(ind, fn))