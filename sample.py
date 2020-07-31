# -*- coding: utf-8 -*-
"""
Created on Thu Mar 19 22:02:02 2020

@author: Leibniz
"""


#tif=misc.imread('./data/sub1.tif')
#misc.imsave('./data/sub1_scipy.tif', tif)
import cv2
tif=cv2.imread('./data/sub8.tif',-1)
#第二个参数是通道数和位深的参数，
#IMREAD_UNCHANGED = -1#不进行转化，比如保存为了16位的图片，读取出来仍然为16位。
#IMREAD_GRAYSCALE = 0#进行转化为灰度图，比如保存为了16位的图片，读取出来为8位，类型为CV_8UC1。
#IMREAD_COLOR = 1#进行转化为RGB三通道图像，图像深度转为8位
#IMREAD_ANYDEPTH = 2#保持图像深度不变，进行转化为灰度图。
#IMREAD_ANYCOLOR = 4#若图像通道数小于等于3，则保持原通道数不变；若通道数大于3则只取取前三个通道。图像深度转为8位
tig=tif[:,:,0:3]
img = cv2.normalize(tig, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
img = img[...,[2,1,0]]
cv2.imwrite('./test.jpg',img)
'''
cv2.imshow("hahah",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
#tif = TIFF.open('./data/sub2.tif', mode='r').read_image()