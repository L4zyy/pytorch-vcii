"""This module takes care of motion estimation and output to the format that is compatible with VCII"""
# Using skvideo for block estimation
import skvideo.io
import skvideo.motion
import skvideo.datasets
import skvideo

from PIL import Image
import numpy as np
from time import time

from numpy.lib.stride_tricks import as_strided

def tile_array(a, b0, b1):
    """Upsample image with nearest neighbor refering https://stackoverflow.com/questions/32846846/quick-way-to-upsample-numpy-array-by-nearest-neighbor-tiling
    
    Arguments:
        a {np.array} -- Image to be upsampled
        b0 {int} -- blocksize h
        b1 {[int]} -- blocksize w
    
    Returns:
        np.array -- upsampled image
    """    
    r, c = a.shape                                    # number of rows/columns
    rs, cs = a.strides                                # row/column strides 
    x = as_strided(a, (r, b0, c, b1), (rs, 0, cs, 0)) # view a as larger 4D array
    return x.reshape(r*b0, c*b1)  

def load_video_data(filename):
    """load video data for skvideo
    
    Arguments:
        filename {string} -- The name of the video file
    
    Returns:
        videodata {} -- The video data compatible with skvideo
    """
    videodata = skvideo.io.vread(filename)
    return videodata

def get_motion_xy(videodata, bs=8, method="DS"):
    """get motion estimates on two greyscale images representin x and y. Video size must be multiples of block size

    Arguments:
        videodata {nparray} -- The video data compatible with skvideo

    Keyword Arguments:
        bs {int} -- block size bs * bs  (default: {8})
        method {str} -- block motion estimation algorithm to use; Check skvideo for details (default: {"DS"})

    Returns:
        (np.array, np.array) -- x and y motion estimates
    """    
    motion = skvideo.motion.blockMotion(videodata,method=method, mbSize=bs)

    # Size of the picture
    image_size = (motion.shape[1]*bs, motion.shape[2]*bs)
    # Size of the whole vidimage_sizelection represented in matrix
    video_shape = (motion.shape[0]+1, image_size[0], image_size[1])

    x_motion = np.zeros(video_shape)
    y_motion = np.zeros(video_shape)
    for f in range(video_shape[0]-1):  # We only want the motion starting from frame 1
        x_motion[f+1,:,:] = tile_array(motion[f, :, :, 1], bs, bs)
        y_motion[f+1,:,:] = tile_array(motion[f, :, :, 0], bs, bs)
    return x_motion, y_motion

def main():
    pass