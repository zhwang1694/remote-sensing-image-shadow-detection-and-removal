# -*- coding: utf-8 -*-
"""
Created on Sat Mar 28 16:58:45 2020

@author: Leibniz
"""

from tkinter import *
from tkinter import filedialog
from PIL import Image,ImageTk
import cv2 as cv
import numpy as np

import gc
import pandas as pd
from skimage.color import lab2lch, rgb2lab
from skimage.exposure import rescale_intensity
from skimage.morphology import disk
from sklearn.cluster import KMeans

class ImageTransfer(object):
    def __init__(self,src=None):
        #图像来源
        self.image=src

    def cv2PIL(self):
        #opencv格式图像转为PIL格式
        pil_image = Image.fromarray(cv.cvtColor(self.image, cv.COLOR_BGR2RGB))
        return pil_image

    def PIL2cv(self):
        #PIL格式图像转换为opencv格式
        cv_image = cv.cvtColor(np.asarray(self.image), cv.COLOR_RGB2BGR)
        return cv_image

    def np2cv(self):
        #将numpy数组转换为opencv格式图像
        return self.image

    def np2PIL(self):
        #numpy数据转换为PIL图像
        pil_image=Image.fromarray(self.image)
        return pil_image

    def tk_image(self):
        #将图像转换为可在tkinter框架中展示的图像
        pil_img = self.cv2PIL()
        show_image = ImageTk.PhotoImage(pil_img)
        return show_image
    
class InitPage(object):
    def __init__(self,master=None):
        self.root=master
        self.width=750
        self.height=400
        self.root.geometry('%dx%d' % (self.width,self.height))
        
        self.defaultImagePath="./sources/src.jpg"
        self.defaultResultPath="./sources/result.jpg"
        self.src = self.tif_read(self.defaultImagePath)
        self.result = self.tif_read(self.defaultResultPath)
        
        self.mask=np.zeros(self.src.shape)
        
        self.createPage()
     
    def createPage(self):
        #在root框架下创建页面
        self.page = Frame(self.root,width=self.width,height=self.height)  # 创建Frame
        self.page.pack()
        #self.threadGenerate()
        
        #text label
        #original_text_label=Label(self.page,text="原始图像",font=("", 11))
        #original_text_label.place(relx=0.1, rely=0.0, relwidth=0.3, relheight=0.15)

        #result_text_label = Label(self.page, text="结果图像",font=("", 11))
        #result_text_label.place(relx=0.54, rely=0.0, relwidth=0.3, relheight=0.15)
        
        
        #image label
        show_image1=self.show_image(self.src)
        self.original_image_label = Label(self.page, image=show_image1)
        self.original_image_label.image=show_image1
        self.original_image_label.place(relx=0.1, rely=0.1, relwidth=0.35, relheight=0.65)

        show_image2 = ImageTransfer(self.normalSize(self.result)).tk_image()
        self.result_image_label = Label(self.page, image=show_image2)
        self.result_image_label.image=show_image2
        self.result_image_label.place(relx=0.55, rely=0.1, relwidth=0.35, relheight=0.65)
        
        button1 = Button(self.page, text='选择图像', command=self.chooseImage)
        button1.place(relx=0.05, rely=0.85, relwidth=0.1, relheight=0.1)
        button2 = Button(self.page, text='阴影检测', command=self.shadowDetection)
        button2.place(relx=0.25, rely=0.85, relwidth=0.1, relheight=0.1)
        button3 = Button(self.page, text='阴影叠加', command=self.shadowAdd)
        button3.place(relx=0.45, rely=0.85, relwidth=0.1, relheight=0.1)
        button4 = Button(self.page, text='阴影去除', command=self.shadowRemoval)
        button4.place(relx=0.65, rely=0.85, relwidth=0.1, relheight=0.1)
        button5 = Button(self.page, text='保存结果', command=self.saveImage)
        button5.place(relx=0.85, rely=0.85, relwidth=0.1, relheight=0.1)
        
        
        #
        
    def shadowDetection(self, convolve_window_size = 5, num_thresholds = 0, struc_elem_size = 3):
        #阴影检测函数
        
        #判断卷积窗口格式
        if (convolve_window_size % 2 == 0):
            raise ValueError('Please make sure that convolve_window_size is an odd integer')
        
        src=self.src.copy()
        img = cv.cvtColor(src, cv.COLOR_BGR2RGB)
        width=img.shape[0]
        
        #阈值计算
        if num_thresholds==0:
            num_thresholds=int(0.01747*width+2.106)
            if num_thresholds<3:
                num_thresholds=3
            if num_thresholds>9:
                num_thresholds=10
        
        #rgb图像转lab域,lab域转lch域
        lch_img = np.float32(lab2lch(rgb2lab(img)))
    
        #归一化
        l_norm = rescale_intensity(lch_img[:, :, 0], out_range = (0, 1))
        h_norm = rescale_intensity(lch_img[:, :, 2], out_range = (0, 1))
        
        #计算sr值
        sr_img = (h_norm + 1) / (l_norm + 1)
        
        #log运算
        log_sr_img = np.log(sr_img + 1)
        
        del l_norm, h_norm, sr_img
        gc.collect()
        
        #平均滤波
        avg_kernel = np.ones((convolve_window_size, convolve_window_size)) / (convolve_window_size ** 2)
        blurred_sr_img = cv.filter2D(log_sr_img, ddepth = -1, kernel = avg_kernel)
        
        #释放内存
        del log_sr_img
        gc.collect()
        
        #图像平展化
        flattened_sr_img = blurred_sr_img.flatten().reshape((-1, 1))
        
        #图像内部像素聚类，得到阴影类
        labels = KMeans(n_clusters = num_thresholds + 1, max_iter = 10000).fit(flattened_sr_img).labels_
        flattened_sr_img = flattened_sr_img.flatten()
        df = pd.DataFrame({'sample_pixels': flattened_sr_img, 'cluster': labels})
        
        #otsu法设定阈值
        threshold_value = df.groupby(['cluster']).min().max()[0]
        df['Segmented'] = np.uint8(df['sample_pixels'] >= threshold_value)
           
        del blurred_sr_img, flattened_sr_img, labels, threshold_value
        gc.collect()
         
        #初始化mask
        shadow_mask_initial = np.array(df['Segmented']).reshape((img.shape[0], img.shape[1]))
        struc_elem = disk(struc_elem_size)
        #形态学滤波
        shadow_mask = np.expand_dims(np.uint8(cv.morphologyEx(shadow_mask_initial, cv.MORPH_CLOSE, struc_elem)), axis = 0)
        
        del df, shadow_mask_initial, struc_elem
        gc.collect()
        
        mask=shadow_mask[0]
        self.mask=mask
        self.result= cv.cvtColor(mask*255, cv.COLOR_GRAY2RGB)
        self.page.destroy()
        self.createPage()
            
        
    
    def show_image(self,img):
        show_image = ImageTransfer(self.normalSize(img)).tk_image()
        return show_image 
        
        
    def chooseImage(self):
        file_path = filedialog.askopenfilename()
        print("图像路径:",file_path)
        self.src=self.tif_read(file_path)
        if self.src is None:
            print("路径中不能含有中文字符!")
            self.src = self.tif_read(self.defaultImagePath)
            self.result = self.tif_read(self.defaultResultPath)
        else:
            self.result = self.tif_read(self.defaultResultPath)
            self.page.destroy()
            self.createPage()
            print("载入图像中...")
    
    def shadowAdd(self):
        src=self.src.copy()
        img = cv.cvtColor(src, cv.COLOR_BGR2RGB)
        mask=self.mask
        #阴影叠加
        _,contours,hierarchy = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        img = img[:, :, ::-1]
        img[..., 2] = np.where(mask == 1, 255, img[..., 2])
        #cv.imwrite("img_mask.jpg",img)
        self.result=img
        self.page.destroy()
        self.createPage()
        
    #阴影去除函数
    def shadowRemoval(self,exponent = 1):
        src=self.src.copy()
        img = cv.cvtColor(src, cv.COLOR_BGR2RGB)
        shadow_mask=self.mask
        #print(shadow_mask.shape)
        
        corrected_img = np.zeros((img.shape), dtype = np.uint8)
        non_shadow_mask = np.uint8(shadow_mask == 0)
        
        
        for i in range(img.shape[2]):
            #阴影区域选择
            shadow_area_mask = shadow_mask * img[:, :, i]
            #非阴影区域选择
            non_shadow_area_mask = non_shadow_mask * img[:, :, i]
            
            #阴影状态输出：按照论文的公式计算结果
            shadow_stats = np.float32(np.mean(((shadow_area_mask ** exponent) / np.sum(shadow_mask))) ** (1 / exponent))
            non_shadow_stats = np.float32(np.mean(((non_shadow_area_mask ** exponent) / np.sum(non_shadow_mask))) ** (1 / exponent))
            #计算非阴影部分与阴影部分的比例q
            mul_ratio = ((non_shadow_stats - shadow_stats) / shadow_stats) + 1
            #最终得到的图像
            corrected_img[:, :, i] = np.uint8(non_shadow_area_mask + np.clip(shadow_area_mask * mul_ratio, 0, 255))
        self.result=cv.cvtColor(corrected_img, cv.COLOR_RGB2BGR)
        self.page.destroy()
        self.createPage()
    
    def saveImage(self):
        file_path = filedialog.asksaveasfilename(title=u'保存文件')
        print("保存路径:",file_path)
        cv.imwrite(file_path,self.result)
        #cv.imwrite(file_path,cv.cvtColor(self.result,cv.COLOR_RGB2BGR))
    
    def normalSize(self,img):
        return cv.resize(img,(260,260))
    
    def tif_read(self,image_file):
        tif=cv.imread(image_file,-1)
        if tif.shape[-1]==4:
            tif_rgb=tif[:,:,0:3]
            src = cv.normalize(tif_rgb, dst=None, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
            src = src[...,[2,1,0]]
            src=np.array(src,dtype='uint8') 
        else:
            src=tif
        #cv.imwrite("./src.jpg",src)
        return src
        
    def threadGenerate(self):
        th=threading.Thread(target=self.xx())
        th.setDaemon(True)
        th.start()
        
    
    
    
    
if __name__=='__main__':
    root=Tk()
    root.title("去阴影系统")
    InitPage(root)
    root.mainloop()
