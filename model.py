
import hytraj
import glob
import numpy as np

def get_model(model_dir='/home/expai/project/tdump3', callsign=None):
    if callsign:
        flist = glob.glob(f'{model_dir}/*{callsign}*')
    else:
        flist = glob.glob(f'{model_dir}/tdump*txt')
    #print(flist)
    #for f in flist:
    #    print(f)
    #    hytraj.open_dataset(f)
    # callsign + delay
    taglist = [x.split('.')[-4] + '.' + x.split('.')[-2] for x in flist]

    #print(taglist)
    if flist:
        df = hytraj.combine_dataset(flist,taglist=taglist)
        df['balloon_callsign'] = df['pid'].str.split('.').str[0]
        df['delay'] = df['pid'].str.split('.').str[1]
        df['delay'] = df['delay'].str.replace('d','').astype(int)
        return df

def check_files(model_dir='/home/expai/project/tdump', callsign=None):
    if callsign:
        flist = glob.glob(f'{model_dir}/*{callsign}*')
    else:
        flist = glob.glob(f'{model_dir}/tdump*txt')
    for x in flist:
        try:
            df = hytraj.open_dataset(x)
        except Exception as e:
            print(f'Error with {x}: {e}')
        # check length of time
        max_traj_age = df['traj_age'].unique()
        max_traj_age = np.max(max_traj_age)
        if max_traj_age < 336: 
            print(f'Warning: {x} has traj_age {max_traj_age}')
