import sys

if sys.version_info[0] < 3 and sys.version_info[1] < 2:
    raise Exception("Must be using >= Python 3.2")

from os import path
import os
from tqdm import tqdm
from glob import glob
from moviepy.editor import *


args = {
    'data_root': '/jumbo/jinlab/datasets/lrs3/trainval/',
    'ngpu': 2,
    'preprocessed_root': './LRS3_preprocess/train',
    'batch_size': 256
}

def process_audio_file(vfile, args):
    vidname = os.path.basename(vfile).split('.')[0]
    dirname = vfile.split('/')[-2]

    fulldir = path.join(args['preprocessed_root'], dirname, vidname)
    os.makedirs(fulldir, exist_ok=True)

    wavpath = path.join(fulldir, 'audio.wav')
    
    # Extracting audio using moviepy
    clip = VideoFileClip(vfile)
    clip.audio.write_audiofile(wavpath)
    clip.close()

def main(args):
    print('Started processing for {}'.format(args['data_root']))

    filelist = glob(path.join(args['data_root'], '*/*.mp4'))

    print('Dumping audios...')
    for vfile in tqdm(filelist):
        try:
            process_audio_file(vfile, args)
        except KeyboardInterrupt:
            exit(0)
        except Exception as e:
            print(f"Error processing {vfile}: {e}")
            continue

if __name__ == '__main__':
    main(args)