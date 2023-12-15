# -*- coding: utf-8 -*-
import c3d
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import glob 
from pathlib import Path

class emts(object):
    def __init__(self, path2emts = None):
        if not path2emts: 
            self.path2emts = '../../data/0409'
            self.seeker_emts = self.path2emts + '/*.emt'
        else: 
            self.path2emts = path2emts
            self.seeker_emts = self.path2emts + '/**/*.emt'
        
        
        # self.list_emts = glob.glob(self.seeker_emts)
        self.list_emts = glob.glob(self.seeker_emts, recursive=True)
    def get_emt_list(self, ):
        self.list_emts_paths = [Path(x) for x in self.list_emts ]
        # self.list_emts_base = [x.split('/data/0409') for x in self.list_emts]
        # self.list_emts_split = [x.split('/data/0409') for x in self.list_emts]
        self.list_emts_names = [x.name for x in self.list_emts_paths]
        self.list_emts_stems = [x.stem for x in self.list_emts_paths]        
        return self.list_emts
    def print_emts_stems(self,):
        # [print(f' {x} ''\n') for x in self.list_emts_stems]        
        # [print(f'\item {x} ''\n') for x in self.list_emts_stems]        
        [print(f'{x} ') for x in self.list_emts_stems]        
    # def 
    def get_emt_dict(self):
        self.emt_dict = {}
        pass
    
def scatter3d_matplot(dataframe_):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    # n = 100
    
    # # For each set of style and range settings, plot n random points in the box
    # # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    # # for m, zlow, zhigh in [('o', -50, -25), ('^', -30, -5)]:
    # xs = randrange(n, 23, 32)
    # ys = randrange(n, 0, 100)
    # zs = randrange(n, zlow, zhigh)
    ax.scatter(dataframe_['X'], dataframe_['Y'], dataframe_['Z'], )
               # marker=m)
    
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
def load_from_emt_3Dtracks(emt_file_path):
    pass
def load_from_emt_first(emt_file_path):
    df0 = pd.read_csv(emt_file_path, skiprows=9, delimiter=r"\s+")
    df = df0.iloc[:,2:]
    df2 = pd.DataFrame(df.values.reshape(89,3), columns = ['X', 'Y','Z'])
    return df2
def c3d_reading(c3d_file_path):
    with open(c3d_file_path, 'rb') as handle:
        reader = c3d.Reader(handle)
        # for i, (points, analog) in enumerate(reader.read_frames()):
        for i, (points, analog) in enumerate(reader.read_frames()):
            print('Frame {}: {}'.format(i, points.round(2)))
def c3d_reading_2(c3d_file_path):      
    # for _ in reader.read_frames(): print(_)
    reader = c3d.Reader(open(c3d_file_path, 'rb'))
    for i, points, analog in reader.read_frames():
        print('frame {}: point {}, analog {}'.format(i, points.shape, analog.shape))

    for i, points in reader.read_frames():
        print(f'{i}')
    
    for _ in dir(reader): 
        # print(f'*++++ \n { _ }')
        print(f'{ _ }')
        # for __ in dir(_): 
            # print(f'***** \n { __ }')
            
    r1 = reader.first_frame
    r2 = reader.point_labels
    g_keys = reader.group_keys()
    rf = reader.read_frames
    g_val = reader.group_values
    print(f'{r1} , {r2}, {g_keys}, {rf}, {g_val}')
    
    
