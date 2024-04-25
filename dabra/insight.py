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
            self.path2emts = '../../data/0409/'
            self.seeker_emts = self.path2emts + '/*.emt'
        else: 
            self.path2emts = path2emts
            self.seeker_emts = self.path2emts + '/**/*.emt'
            print('Check for multiple conditions... ')
        
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
        self.get_emt_list()
        # [print(f' {x} ''\n') for x in self.list_emts_stems]        
        # [print(f'\item {x} ''\n') for x in self.list_emts_stems]        
        [print(f'{x} ') for x in self.list_emts_stems]        
    # def 
    def get_emt_dict(self):
        self.emt_dict = {}
        emt_file_path = f'{self.path2emts}3D Point Tracks.emt' # :dart:: putit to conf
        u1d_file_path = f'{self.path2emts}1D Point Tracks.emt'
        vol_file_path = f'{self.path2emts}volume tracks.emt'
        v1d_file_path = f'{self.path2emts}1D velocity track.emt' # :dart:: putit to conf
        c1d_file_path = f'{self.path2emts}Scalar Cycle Sequences.emt'
        scl_file_path = f'{self.path2emts}Scalar Tracks.emt'
        
    
        
        df_3DU = pd.read_csv(emt_file_path, skiprows=9, delimiter=r"\s+")
        df_scl = pd.read_csv(scl_file_path, skiprows=9, delimiter=r"\s+")
        
        try:
            df_Vol = pd.read_csv(vol_file_path, skiprows=9, delimiter=r"\s+")
        except:
            df_Vol = pd.DataFrame()
        try:
            df_1DU = pd.read_csv(u1d_file_path, skiprows=9, delimiter=r"\s+")
        except:
            df_1DU = pd.DataFrame()
        
        try:
            df_1DV  = pd.read_csv(v1d_file_path, skiprows=9, delimiter=r"\s+")
        except: 
            df_1DV  = pd.DataFrame()
            
        try:
           df_Cyc  = pd.read_csv(c1d_file_path, skiprows=7, delimiter=r"\s+")
        except:
           df_Cyc  = pd.DataFrame()
        
        # df_3DU = df_3DU.dropna(axis = 0) correct ??
        # df_3DU = df_3DU.dropna(axis = 1) 
        # df_3DU = df_3DU.dropna(axis = 1) 
        
        df_1DU = df_1DU.dropna(axis = 1)
        df_Vol = df_Vol.dropna(axis = 1)
        
        # soft preprocessing... getting rid off frame ... its in index
        # df_.columns[2::]
        # spravny naming columns 
        
        
        self.emt_dict['3DU'] = df_3DU
        self.emt_dict['Scl'] = df_scl
        self.emt_dict['1DU'] = df_1DU
        self.emt_dict['1DV'] = df_1DV
        self.emt_dict['Vol'] = df_Vol
        self.emt_dict['Cyc'] = df_Cyc
        
        
        
        
        def get_emt_support_summary(self):
            self.emt_summary_dict = {}
            pass
        
        return self.emt_dict
    def overview_plotting(self,):
        # for self.emt_dict
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
    
    
