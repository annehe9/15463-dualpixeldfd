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

#taken out of optimization to generate diagram
def parameterized_blur_kernel(kradius, maxksize):
		circle = np.zeros((maxksize, maxksize))
		#y,x = np.ogrid[-kradius: kradius+1, -kradius: kradius+1]
		radius = abs(kradius)
		y,x = np.ogrid[-maxksize//2: maxksize//2, -maxksize//2: maxksize//2]
		diskmask = x**2+y**2 <= kradius**2
		circle[diskmask] = 1
		#io.imshow(circle)
		#io.show()
		disk_ker = np.zeros((maxksize,maxksize)) # accumulate translating disk kernel
		for i in np.arange(0, 2*radius+2):
			translate = np.identity(3)
			translate[0,2] = np.sign(kradius)*i
			t = transform.ProjectiveTransform(translate)
			warped = transform.warp(circle, t.inverse)
			#io.imshow(warped)
			#io.show()
			disk_ker = disk_ker + circle * warped
		output = 0.5*disk_ker/np.sum(disk_ker) # scale
		io.imshow(output)
		io.show()
		return output

parameterized_blur_kernel(-25,100)