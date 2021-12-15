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

def solve_translating_disk_kernel(maxksize, lpatch, rpatch, border):

	def shift_direction(): 
	# sign of shift can be determined by minimizing index for windowed sum of squared differences
		cols = lpatch.shape[1]
		im1 = lpatch[:,cols%4:]
		im2 = rpatch[:,cols%4:]
		r = cols//4
		win = r*2
		ssd = np.zeros(win+1)
		for i in range(win+1):
			ssd[i] = np.sum(np.sum( (im1[:,r:r+win] - im2[:,i:i+win])**2 ))
		minind = np.argmin(ssd)
		if minind > r:
			return -1
		else:
			return 1

	def loss_fun(kradius):
		h = parameterized_blur_kernel(kradius, maxksize)
		l = signal.convolve2d(lpatch, np.fliplr(h), mode='same')
		r = signal.convolve2d(rpatch, h, mode='same')

		l = l[border:l.shape[0]-border, border:l.shape[1]-border]
		r = r[border:r.shape[0]-border, border:r.shape[1]-border]

		err = (l-r)/255
		xerr = np.mean(err**2)
		return xerr

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
		#io.imshow(output)
		#io.show()
		return output

	#parameterized_blur_kernel(4,21)

	lb = -(maxksize-1)/2;
	ub = (maxksize-1)/2;
	x0 = shift_direction()
	
	result = scipy.optimize.minimize(loss_fun, [x0], bounds=[(lb,ub)], method='Powell') 
		#options={'eps':1e-20, 'maxfun':2500, 'maxiter':2500})
	#result = scipy.optimize.minimize_scalar(loss_fun, method='bounded', bounds=(lb,ub))
	if not result.success:
		print("Optimization failed.")
		print(result.message)
		return
	x = result.x
	fval = result.fun
	k_est = parameterized_blur_kernel(x, maxksize)

	return x, k_est, fval
