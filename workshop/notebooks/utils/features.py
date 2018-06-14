from scipy.misc import imread, imresize
from skimage.measure import regionprops
from scipy.spatial import cKDTree
import numpy as np
from math import isinf



feature_list = ['area', 'convex_area', 'eccentricity', 
                'intense_area', 'convection_area',
                'convection_stratiform_ratio', 'intense_stratiform_ratio',
                'intense_convection_ratio', 'mean_intensity', 'max_intensity',
                'intensity_variance', 'major_axis_length', 'minor_axis_length',
                'solidity', 'filename']
                
def special_reshape(img, xs, ys):

    max_dim = np.argmax(img.shape)
    min_dim = np.argmin(img.shape)
    
    diff = int((img.shape[max_dim] - img.shape[min_dim]) / 2)

    if max_dim == 0:
        pad_arr = ((0,0), (diff,diff))
    else:
        pad_arr = ((diff,diff), (0,0))

    im = np.pad(img, pad_arr, mode='constant')

    return imresize(im, (xs, ys))


def calc_features(fn=None, props=None, stratiform=4, convection=8, intense=10, 
                  pixel_area=4, pixel_length=2, dbz_compression=5):
    
    if props is None:
        img = imread(fn, mode='P')
        props = regionprops(1*(img>0), intensity_image=img)[0]
    else:
        img = props.intensity_image
        
    area = props.area * pixel_area
    convex_area = props.convex_area * pixel_area

    eccentricity = props.eccentricity
    intense_area = np.sum(props.intensity_image >= intense) * pixel_area
    convection_area = np.sum(props.intensity_image >= convection) * pixel_area
    stratiform_area = np.sum(props.intensity_image >= stratiform) * pixel_area
    
    if stratiform_area > 0 and intense_area > 0:
        convection_stratiform_ratio = convection_area / stratiform_area
        intense_stratiform_ratio = intense_area / stratiform_area
        intense_convection_ratio = intense_area / convection_area
    else:
        convection_stratiform_ratio = 0
        intense_stratiform_ratio = 0
        intense_convection_ratio = 0

    mcs_probability = 0.0
    tropical_probability = 0.0
    synoptic_probability = 0.0
    ucc_probability = 0.0
    clutter_probability = 0.0
    
    mean_intensity = props.mean_intensity * dbz_compression
    max_intensity = props.max_intensity * dbz_compression
    
    major_axis_length = props.major_axis_length * pixel_length
    minor_axis_length = props.minor_axis_length * pixel_length
    
    ind = np.where(props.intensity_image>0)
    intensity_variance = np.var(props.intensity_image[ind] * dbz_compression) 
    
    arr = np.array([area, convex_area, eccentricity, 
                    intense_area, convection_area,
                    convection_stratiform_ratio, intense_stratiform_ratio,
                    intense_convection_ratio, mean_intensity, max_intensity,
                    intensity_variance, major_axis_length, minor_axis_length,
                    props.solidity, fn])

    feature_dict = dict((key, value) for (key, value) in zip(feature_list, arr))
        
    return feature_dict
