# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:43:58 2020

@author: pc
"""

# =============================================================================
#  图像缩放
#  图像灰度
#  图像二值化
#  联通区域
#  形态学变换
#  
#
#
#
#
#
#
#
#
# =============================================================================

import cv2
import os




class ImageProcess(object):
    """
    这是一个图像处理类
    
    """
    def __init__(self,images_file):
        
        self.image_file  = images_file
        
        # 判断是文件还是文件夹
        if os.path.isdir(images_file):
            print('input is directory')
            self.image_file  = images_file
            self.isfile = False
        elif os.path.isfile(images_file):
            print('input is file')
            self.isfile = True
            self.image_name = images_file
            
        else:
            print('plz input a path  or a file')

        
        

        
    def resize(self):
        pass
    
    def forward(self,threshvalues,dst_file):
        # 当输入的路径是图片时
        if self.isfile:
            img = cv2.imread(self.image_name)
            image_name = self.image_name
            thresh = Threshold(img,image_name,dst_file )
            thresh.forward(threshvalues)
        # 当输入的路径是文件夹    
        else:
            for image_name in os.listdir(self.image_file):
                print(image_name)
                img = cv2.imread(os.path.join(self.image_file,image_name))
                thresh = Threshold(img,image_name,dst_file )
                print('threshold is  {}'.format(threshvalues))
                thresh.forward(threshvalues=threshvalues)
        



class Threshold(object):
    '''
    阈值处理工作， 包括 设定阈值，自动阈值
    '''
    def __init__(self,img,image_name,dst_root):
        super(Threshold,self).__init__()
        
        self.img = img
        self.gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        self.image_name = image_name
        
        self.dst_root = dst_root
        
        if not os.path.exists(self.dst_root):
            os.makedirs(self.dst_root)
        #self.blur = None 

    def blured(self, size = (5,5)):
        blured = cv2.GaussianBlur(self.img,size,0)
        return blured      
        
    
    def threshold(self,threshvalue,INV= False):
        #threshold(gray_src, dst, threshold_value, threshold_max, THRESH_BINARY);
        if INV:
            ret,binary = cv2.threshold(self.gray, threshvalue, 255, cv2.THRESH_BINARY_INV)
        else:
            ret,binary = cv2.threshold(self.gray, threshvalue, 255, cv2.THRESH_BINARY)
        return binary
    
    
    def truncate(self,threshvalue):
        '''
        截断,高于阈值的像素值变为阈值大小，低于阈值的像素值不变
        '''
        ret,binary = cv.threshold(self.gray,threshvalue, 255, cv2.THRESH_TRUNC)
        return binary
    
    
    def tozero(self,threshvalue,INV = False):
        '''
        置零，低于阈值的像素值变为零，高于阈值的像素值不变
        '''
        if INV:
            ret,binary = cv.threshold(self.gray,threshvalue, 255, cv2.THRESH_TOZERO_INV)
            # 高于阈值的置为零，低于阈值的不变
        else:
            ret,binary = cv.threshold(self.gray,threshvalue, 255, cv2.THRESH_TOZERO)
            
        return binary


    def ostu(self):
        '''
        '''
        ret,binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary
        
        
    def triangle(self):
        '''
        '''
        ret,binary = cv2.threshold(self.gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_TRIANGLE)
        return binary
    
    
    def adaptiveThreshold(self,Block_Size=11,C = 4):
        # cv2.adaptiveThreshold（image,max_BINARY_value,Adaptive_Method,threshold_type,Block_Size,C)
        # max_BINARY_value: 设定的最大灰度值（该参数运用在二进制与反二进制阈值操作中）
        # threshold_type,阈值的类型
        # cv2.ADPTIVE_THRESH_MEAN_C 阈值取自相邻区域的平均值
        # cv2.ADPTIVE_THRESH_GAUSSIAN_C：阈值取值相邻区域的加权和，权重为一个高斯窗口
        #Block Size - 邻域大小（用来计算阈值的区域大小）。表明我们要检查图像的11×11像素区域，而不是像在简单的阈值方法中那样尝试对图像进行全局阈值
        #C - 这就是是一个常数，阈值就等于的平均值或者加权平均值减去这个常数。我们提供一个简单的叫做C的参数。这个值是一个从平均值中减去的整数，使我们可以微调我们的阈值
        thresh_mean = cv2.adaptiveThreshold(self.gray, 255 ,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,Block_Size,C)
        thresh_gaussian = cv2.adaptiveThreshold(self.gray, 255 ,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,Block_Size,C)
        return thresh_mean,thresh_gaussian
    
    def forward(self,blured = False, threshvalues = [127]):
        
        if blured == True:
            self.gray = self.blured(size=(5,5))
        
        binarys_threshold = []
        for threshvalue in threshvalues:
            binary_threshold = self.threshold(threshvalue)
            binarys_threshold.append(binary_threshold)
        
        #ostu    
        binary_ostu = self.ostu()
        # triangle
        binary_triangle= self.triangle()
        # adaptiveThreshold
        binary_mean,binary_gaussian = self.adaptiveThreshold(Block_Size=11,C = 4)
         
        # 保存图像 
        for ii,binary_threshold in enumerate(binarys_threshold):
            # 名字
            binary_threshold_name = self.image_name[:-4]+   '_threshvalue_'+str(threshvalues[ii])+'_' +self.image_name[-4:]            
            cv2.imwrite(os.path.join(self.dst_root,binary_threshold_name),binary_threshold)
        #名字
        binary_ostu_name = self.image_name[:-4]+ '_ostu_' + self.image_name[-4:]    
        cv2.imwrite(os.path.join(self.dst_root,binary_ostu_name),binary_ostu)
        
        binary_triangle_name = self.image_name[:-4]+ '_triangle_' + self.image_name[-4:]    
        cv2.imwrite(os.path.join(self.dst_root,binary_triangle_name),binary_triangle)                    
        
        binary_gaussian_name = self.image_name[:-4]+ '_gaussian_'+ self.image_name[-4:]                        
        cv2.imwrite(os.path.join(self.dst_root,binary_gaussian_name),binary_gaussian)
        
        binary_mean_name = self.image_name[:-4]+ '_adaptive_mean_'+ self.image_name[-4:]                        
        cv2.imwrite(os.path.join(self.dst_root,binary_mean_name),binary_mean)

        
        
    
    
    
    
    

class Morphological_Transform(ImageProcess):
    """
    形态学变换类, 输入一张图片，输出图片的各种形态学处理后的图像
    """
    def __init__(self,img,kernel_sizes=[(3,3),(5,5)],iteraions = 1):
        super(Morphological_Transform,self).__init__()
        
        self.img = img
        self.kernel_sizes = kernel_sizes
        self.iterations = 1
        
    def binary(self):
        
        return binary
    
    def dilate(self,kernel_size,threshMap):
        # 膨胀
        kernel = np.ones(kernel_size, dtype=np.uint8)
        dilate = cv2.dilate(threshMap, kernel, 1)
        
        return dilate
    
    
      
    def erosion(self,kernel_size,threshMap):
        # 腐蚀
        kernel = np.ones((3, 3), dtype=np.uint8)
        erosion = cv2.erode(threshMap, kernel, iterations=1)
        
        return erosion
    
    
    #开运算：先腐蚀，再膨胀 闭运算：先膨胀，再腐蚀  
    def opening(self,kernel_size,threshMap):
        # 开运算
        kernel = np.ones((5, 5), dtype=np.uint8)
        opening = cv2.morphologyEx(threshMap, cv2.MORPH_OPEN, kernel, 1)
        
        return opening
    
    
    def closing(self,kernel_size,threshMap):
        # 闭运算
        kernel = np.ones((5, 5), dtype=np.uint8)

        closing = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)  ## 有缺陷，填补缺陷
        return closing
    
        
    def process(self):
        
        for kernel_size in  self.kernel_sizes:
            
        
        
        
        pass
    
    def connectedComponentsAnalysis(self,connectivity=8):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_.astype('uint8'), connectivity=8)
        
        pass
    
    
    
    

def main():    
    
    
    
    image_file = 'E:/Code/My_own_project/ultrasound/data/1st/images'
    
    #image_file = 'E:/Code/My_own_project/ultrasound/data/1st/images_circle_out'
    dst_file = 'E:/Code/My_own_project/ultrasound/code/saliency/result/my_ImageProcess_images'
    
    if not os.path.exists(dst_file):
        os.makedirs(dst_file)
    
    for image_name in os.listdir(image_file ):
        print(image_name) 
        img = cv2.imread(os.path.join(image_file,image_name))
        thresh = Threshold(img,image_name,dst_file )
        thresh.forward(threshvalues = [10,20,30,40,50,60,70,80,90,127])

    
    
   imageprocess = ImageProcess(image_file)
   imageprocess.forward([10,20,30,40,50,60,70,80,90,127],dst_file)
    
    
    
    
    
    
    
    
    