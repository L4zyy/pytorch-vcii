import numpy as np
import youtube_dl
import subprocess
import sys
import logging

class MyLogger(object):
    def debug(self, msg):
        pass
    
    def warning(self, msg):
        pass
    
    def error(self, msg):
        print(msg)

class FrameExtractor():
    def __init__(self, output_dir, start_point):
        self.url_starter = 'youtube.com/watch?v='
        self.output_dir = output_dir
        self.v_info = None # v_id, start, end
        self.counter = start_point

        self.category = output_dir.split('/')[-2]
        logging.basicConfig(filename='{}_{}.log'.format(self.category, start_point))
        
    def my_hook(self, d):
        if d['status'] == 'finished':
            print('[{}]Done downloading, now converting ...'.format(self.counter))
            self.counter += 1
            subprocess.call(['ffmpeg', \
                             '-loglevel', \
                             'warning', \
                             '-ss', \
                             self.v_info[1], \
                             '-i', \
                             '{}{}_{}_{}.mp4'.format(self.output_dir, *self.v_info), \
                             '-vframes', \
                             '97', \
                             '-vf', \
                             'scale=352:288', \
                             '{}{}_{}_{}_%04d.png'.format(self.output_dir, *self.v_info) \
                            ])
            print('Done converting.')
            

    def extract_frames(self, info):
        self.v_info = info

        ydl_opts = {
            'format': 'bestvideo[fps<=30]',
            'outtmpl': '{}%(id)s_{}_{}.%(ext)s'.format(self.output_dir, self.v_info[1], self.v_info[2]),
            'logger': MyLogger(),
            'progress_hooks': [self.my_hook],
        }

        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            try:
                ydl.download([self.url_starter + self.v_info[0]])
            except Exception:
                logging.exception('download video [{}]({}) failed.'.format(self.counter, self.v_info[0]))
            else:
                subprocess.call(['rm', \
                 '{}{}_{}_{}.mp4'.format(self.output_dir, *self.v_info) \
                ])


if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python kinetics_dataset_downloader.py [data_list_path] [output_dir] [start] [end]')
        exit()
    data_list_path = sys.argv[1]
    output_dir = sys.argv[2]
    start = sys.argv[3]
    end = sys.argv[4]

    texts = np.genfromtxt(data_list_path,dtype='str')
    data = []
    for t in texts:
        data.append(t.rsplit('_', 2)) # [id, start, end]

    ext = FrameExtractor(output_dir, int(start))
    for d in data[int(start)-1:int(end)-1]:
        ext.extract_frames(d)
        break
