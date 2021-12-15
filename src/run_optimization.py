import time
import os
import skimage
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import scipy
import glob
from skimage.transform import rescale
from skimage.color import rgb2gray
from tqdm import tqdm
from skimage import transform
from scipy import signal
from scipy import ndimage
from optimization import solve_translating_disk_kernel
from skimage import img_as_ubyte

save_path = "../data/results/"
imgname = '0002' # should have imgname_B.png, imgname_L.png, and imgname_R.png
resize_val = 0.3 # if img too large                
patchsize = 111 # odd
maxksize = 41 # odd          
stride = 33 # odd
border = 25 # > half of maxksize
settings = imgname + "_p_" + str(patchsize) + "_k_" + str(maxksize) + "_s_" + str(stride) + "_b_" + str(border) + "_r_" + str(resize_val)

def run_translating_disk_kernel(patchsize, maxksize, limg, rimg, cimg, border, stride):
	shape = limg.shape
	m = (patchsize-1)//2 # half of patch
	mids = (stride-1)//2 # half of stride

	rowsteps = ( (shape[0]-m) - m )// stride
	colsteps = ( (shape[1]-m) - m )// stride

	rowspace = np.arange(m, shape[0]-m, stride).astype("int")
	colspace = np.arange(m, shape[1]-m, stride).astype("int")
	#rowspace = np.linspace(m+1, m+1+stride*rowsteps, rowsteps+1).astype("int")
	#colspace = np.linspace(m+1, m+1+stride*colsteps, colsteps+1).astype("int")

	depthmap = np.zeros(shape)
	errmap = np.zeros(shape)
	sobelmap = np.zeros(shape)

	counter = 1
	total_time = 0
	# plt.ion()
	# fig = plt.figure()
	# ax1 = fig.add_subplot(221)
	# ax2 = fig.add_subplot(222)
	# ax3 = fig.add_subplot(223)
	# ax4 = fig.add_subplot(224)
	for i in rowspace:
		print(counter,"of",rowsteps+1)
		start = time.time()
		for j in colspace:
			lpatch = limg[i-m:i+m,j-m:j+m]
			rpatch = rimg[i-m:i+m,j-m:j+m]
			cpatch = cimg[i-m:i+m,j-m:j+m]

			sobel_val = ndimage.sobel(cpatch)
			sobel_val = sobel_val[2:sobel_val.shape[0]-2, 2:sobel_val.shape[1]-2]
			sobel_val = np.mean(np.abs(sobel_val))

			kradius, k_est, fval = solve_translating_disk_kernel(maxksize,lpatch,rpatch,border)

			depthmap[i-mids:i+mids,j-mids:j+mids] = kradius
			errmap[i-mids:i+mids,j-mids:j+mids] = fval
			sobelmap[i-mids:i+mids,j-mids:j+mids] = sobel_val
			
			# ax1.set_title('Original')
			# ax1.imshow(limg)
			# ax1.set_axis_off()

			# ax2.set_title('Coarse depth map')
			# ax2.imshow(depthmap)
			# ax2.set_axis_off()

			# ax3.set_title('Inverse error')
			# ax3.imshow(1.0/errmap)
			# ax3.set_axis_off()

			# ax4.set_title('Sobel value')
			# ax4.imshow(sobelmap)
			# ax4.set_axis_off()
			
			# fig.canvas.draw()
			# fig.canvas.flush_events()
		end = time.time()
		total_time += end-start
		print("Took", (end-start)/60, "minutes. Estimated time remaining: ", (total_time/counter)*(rowsteps+1-counter)/60)
		counter+=1
	return depthmap, errmap, sobelmap

imgpath = os.path.join("../data/", imgname)
limg = rescale(rgb2gray(io.imread(imgpath+'_L.png')), resize_val)
rimg = rescale(rgb2gray(io.imread(imgpath+'_R.png')), resize_val)
cimg = rescale(rgb2gray(io.imread(imgpath+'_B.png')), resize_val)

depthmap, errmap, sobelmap = run_translating_disk_kernel(patchsize, maxksize, limg, rimg, cimg, border, stride)

shape = cimg.shape
m = (patchsize-1)//2 # half of patch
mids = (stride-1)//2 # half of stride

rowsteps = ( (shape[0]-m) - m )// stride
colsteps = ( (shape[1]-m) - m )// stride

rowspace = np.arange(m, shape[0]-m, stride).astype("int")
colspace = np.arange(m, shape[1]-m, stride).astype("int")

cimg = rescale(io.imread(imgpath+'_B.png'), (resize_val, resize_val, 1))

cimgcropped = cimg[m-mids:m+stride*rowsteps+mids, m-mids:m+stride*colsteps+mids]
depthcropped = depthmap[m-mids:m+stride*rowsteps+mids, m-mids:m+stride*colsteps+mids]
errcropped = errmap[m-mids:m+stride*rowsteps+mids, m-mids:m+stride*colsteps+mids]
print(np.min(errcropped))
print(np.max(errcropped))
sobelcropped = sobelmap[m-mids:m+stride*rowsteps+mids, m-mids:m+stride*colsteps+mids]
sobelcropped = sobelcropped/np.max(sobelcropped)
confidencecropped = np.exp(-1000000)*errcropped
confidence = confidencecropped*sobelcropped
print(np.min(confidence))
print(np.max(confidence))
confidence = (confidence-np.min(confidence)) / (np.max(confidence)-np.min(confidence))
confidence = np.where(confidence == 0, 10^-150, confidence)
confidence = np.clip(confidence, 0, 1)

fullpath = os.path.join(save_path, settings)
os.mkdir(fullpath)
depthcropped = (depthcropped - np.min(depthcropped)) / (np.max(depthcropped) - np.min(depthcropped))
io.imsave(fullpath+"/coarsemap.png", img_as_ubyte(depthcropped))
#io.imsave(fullpath+"/confidence.png", confidence)
io.imsave(fullpath+"/reference.png", cimgcropped)
np.savez(fullpath+"/optresults.npz", depth=depthcropped, confidence=confidence, reference=cimgcropped)
print(fullpath)