import cv2
import gc
import numpy as np
import pandas as pd
#import rasterio
from skimage.color import lab2lch, rgb2lab
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from sklearn.cluster import KMeans

def tif_read(image_file):
    tif=cv2.imread(image_file,-1)
    if tif.shape[-1]==3:
        src=tif
    else:
        tif_rgb=tif[:,:,0:3]
        src = cv2.normalize(tif_rgb, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        src = src[...,[2,1,0]]
        src=np.array(src,dtype='uint8') 
    cv2.imwrite("./src.jpg",src)
    return src
    

def shadow_detection(image_file, shadow_mask_file, convolve_window_size = 5, num_thresholds = 3, struc_elem_size = 3):
    """
    This function is used to detect shadow - covered areas in an image, as proposed in the paper 
    'Near Real - Time Shadow Detection and Removal in Aerial Motion Imagery Application' by Silva G.F., Carneiro G.B., 
    Doth R., Amaral L.A., de Azevedo D.F.G. (2017)
    
    Inputs:
    - image_file: Path of image to be processed for shadow removal. It is assumed that the first 3 channels are ordered as
                  Red, Green and Blue respectively
    - shadow_mask_file: Path of shadow mask to be saved
    - convolve_window_size: Size of convolutional matrix filter to be used for blurring of specthem ratio image
    - num_thresholds: Number of thresholds to be used for automatic multilevel global threshold determination
    - struc_elem_size: Size of disk - shaped structuring element to be used for morphological closing operation
    
    Outputs:
    - shadow_mask: Shadow mask for input image
    
    """
    
    if (convolve_window_size % 2 == 0):
        raise ValueError('Please make sure that convolve_window_size is an odd integer')
        
    #buffer = int((convolve_window_size - 1) / 2)
    
    src=tif_read(image_file)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
    a=img.shape[0]
    num_thresholds=int(0.01747*a+2.106)
    if num_thresholds<3:
        num_thresholds=3
    if num_thresholds>9:
        num_thresholds=10
    
#    with rasterio.open(image_file) as f:
#        metadata = f.profile
#        img = rescale_intensity(np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0]), out_range = 'uint8')
#        img = img[:, :, 0 : 3]
    
    
    lch_img = np.float32(lab2lch(rgb2lab(img)))
    
    
    l_norm = rescale_intensity(lch_img[:, :, 0], out_range = (0, 1))
    h_norm = rescale_intensity(lch_img[:, :, 2], out_range = (0, 1))
    sr_img = (h_norm + 1) / (l_norm + 1)
    log_sr_img = np.log(sr_img + 1)
    
    del l_norm, h_norm, sr_img
    gc.collect()

    

    avg_kernel = np.ones((convolve_window_size, convolve_window_size)) / (convolve_window_size ** 2)
    blurred_sr_img = cv2.filter2D(log_sr_img, ddepth = -1, kernel = avg_kernel)
      
    
    del log_sr_img
    gc.collect()
    
                
    flattened_sr_img = blurred_sr_img.flatten().reshape((-1, 1))
    labels = KMeans(n_clusters = num_thresholds + 1, max_iter = 10000).fit(flattened_sr_img).labels_
    flattened_sr_img = flattened_sr_img.flatten()
    df = pd.DataFrame({'sample_pixels': flattened_sr_img, 'cluster': labels})
    threshold_value = df.groupby(['cluster']).min().max()[0]
    df['Segmented'] = np.uint8(df['sample_pixels'] >= threshold_value)
    
    
    del blurred_sr_img, flattened_sr_img, labels, threshold_value
    gc.collect()
    
    
    shadow_mask_initial = np.array(df['Segmented']).reshape((img.shape[0], img.shape[1]))
    struc_elem = disk(struc_elem_size)
    shadow_mask = np.expand_dims(np.uint8(cv2.morphologyEx(shadow_mask_initial, cv2.MORPH_CLOSE, struc_elem)), axis = 0)
    #shadow_mask*=255
    #print(shadow_mask.shape) #(1,273,386)
    
    del df, shadow_mask_initial, struc_elem
    gc.collect()
    

#    metadata['count'] = 1
#    with rasterio.open(shadow_mask_file, 'w', **metadata) as dst:
#        dst.write(shadow_mask)
    cv2.imwrite(shadow_mask_file,shadow_mask[0])
    
    mask=shadow_mask[0]
    _,contours,hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = img[:, :, ::-1]
    img[..., 2] = np.where(mask == 1, 255, img[..., 2])
    cv2.imwrite("img_mask.jpg",img)
        
    return src,shadow_mask[0]



def shadow_correction(image_file, shadow_mask_file, corrected_image_file, exponent = 1):
    """
    This function is used to adjust brightness for shadow - covered areas in an image, as proposed in the paper 
    'Near Real - Time Shadow Detection and Removal in Aerial Motion Imagery Application' by Silva G.F., Carneiro G.B., 
    Doth R., Amaral L.A., de Azevedo D.F.G. (2017)
    
    Inputs:
    - image_file: Path of 3 - channel (red, green, blue) image to be processed for shadow removal
    - shadow_mask_file: Path of shadow mask for corresponding input image
    - corrected_image_file: Path of corrected image to be saved
    - exponent: Exponent to be used for the calculcation of statistics for unshaded and shaded areas
    
    Outputs:
    - corrected_img: Corrected input image
    
    """
    
#    with rasterio.open(image_file) as f:
#        metadata = f.profile
#        img = rescale_intensity(np.transpose(f.read(tuple(np.arange(metadata['count']) + 1)), [1, 2, 0]), out_range = 'uint8')

    src=tif_read(image_file)
    img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        
#    with rasterio.open(shadow_mask_file) as s:
#        shadow_mask = s.read(1)
    shadow_mask=cv2.imread(shadow_mask_file,0)
    print(shadow_mask.shape)
    
    corrected_img = np.zeros((img.shape), dtype = np.uint8)
    non_shadow_mask = np.uint8(shadow_mask == 0)
    
    
    for i in range(img.shape[2]):
        shadow_area_mask = shadow_mask * img[:, :, i]
        non_shadow_area_mask = non_shadow_mask * img[:, :, i]
        shadow_stats = np.float32(np.mean(((shadow_area_mask ** exponent) / np.sum(shadow_mask))) ** (1 / exponent))
        non_shadow_stats = np.float32(np.mean(((non_shadow_area_mask ** exponent) / np.sum(non_shadow_mask))) ** (1 / exponent))
        mul_ratio = ((non_shadow_stats - shadow_stats) / shadow_stats) + 1
        corrected_img[:, :, i] = np.uint8(non_shadow_area_mask + np.clip(shadow_area_mask * mul_ratio, 0, 255))
    

#    with rasterio.open(corrected_image_file, 'w', **metadata) as dst:
#        dst.write(np.transpose(corrected_img, [2, 0, 1]))

    cv2.imwrite(corrected_image_file,cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR))
        
    return cv2.cvtColor(corrected_img, cv2.COLOR_RGB2BGR)

if __name__=='__main__':
    #sub2、5、6、7效果不好
    #sub1 5 5 3 (273,386)
    #sub2 5 10 3 (416,427)
    #sub3 5 7 3 (202,253)
    #sub4 5 3 3 (63,98)
    #sub5 5 5 3 (190,158)
    #sub6 5 4 3 (120,154)
    #sub7 5 10 3 (479,948)
    #sub8 5 8 3 (269,620)
    image_file='./data/sub1.tif'
    mask_file='./mask.jpg'
    result_file='./result.jpg'
    rgb,mask=shadow_detection(image_file, mask_file)
    corrected=shadow_correction(image_file,mask_file,result_file)
    
    
    cv2.imshow("rgb",rgb)
    cv2.imshow("mask",mask*255)
    cv2.imshow("corrected",corrected)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    