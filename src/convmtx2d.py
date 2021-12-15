import skimage
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import scipy
from scipy.linalg import toeplitz

# based on https://stackoverflow.com/questions/16798888/2-d-convolution-as-a-matrix-matrix-multiplication
# also referred to https://dsp.stackexchange.com/questions/35373/2d-convolution-as-a-doubly-block-circulant-matrix-operating-on-a-vector
# and https://www.cs.uoi.gr/~cnikou/Courses/Digital_Image_Processing/Chapter_04c_Frequency_Filtering_(Circulant_Matrices).pdf
def convmtx2d(F, imshape):
	output_rows = imshape[0] + F.shape[0] - 1
	output_cols = imshape[1] + F.shape[1] - 1
	#print('output shape:', (output_rows, output_cols))

	F_padded = np.pad(F, ((output_rows-F.shape[0], 0),(0, output_cols-F.shape[1])), 'constant', constant_values=0)
	#print(F_padded)

	# Each row -> toeplitz matrix 
	# Number of cols = number of cols of input signal
	toeplitz_list = []
	for i in range(F_padded.shape[0]-1, -1, -1): # last to first row
	    c = F_padded[i, :] # i th row of the F 
	    r = np.zeros(imshape[1])
	    r[0] = c[0] # 0s across top row, first elem same as first of col
	    toeplitz_m = toeplitz(c,r)
	    toeplitz_list.append(toeplitz_m)
	    #print('F '+ str(i)+'\n', toeplitz_m)

	# doubly blocked toeplitz - indices are also a toeplitz matrix
	c = np.arange(1, F_padded.shape[0]+1)
	r = np.zeros(imshape[0])
	r[0] = c[0]
	inds = toeplitz(c, r).astype(int)
	
	toeplitz_shape = toeplitz_list[0].shape # shape of one toeplitz matrix
	h = toeplitz_shape[0]*inds.shape[0]
	w = toeplitz_shape[1]*inds.shape[1]
	doubly_blocked = np.zeros((h,w))

	# tile toeplitz matrices
	for i in range(inds.shape[0]):
	    for j in range(inds.shape[1]):
	        start_i = i * toeplitz_shape[0]
	        start_j = j * toeplitz_shape[1]
	        end_i = start_i + toeplitz_shape[0]
	        end_j = start_j + toeplitz_shape[1]
	        doubly_blocked[start_i:end_i, start_j:end_j] = toeplitz_list[inds[i,j]-1]

	return doubly_blocked

def matrix_to_vector(matrix):
	matrix = np.flipud(matrix)
	matrix = matrix.flatten()
	matrix = matrix[:, np.newaxis]
	return matrix

def vector_to_matrix(vector, mshape):
	vector = np.reshape(vector, mshape)
	vector = np.flipud(vector)
	return vector

def test():
	test = np.array([[1,2,3],[4,5,6]])
	testres = matrix_to_vector(test)
	print(testres)
	print(vector_to_matrix(testres, test.shape))

	#test
	I = np.random.randn(10, 13)
	F = np.random.randn(30, 70)

	convolution_matrix = convmtx2d(F, I.shape)
	vectorI = matrix_to_vector(I)
	output_rows = I.shape[0] + F.shape[0] - 1
	output_cols = I.shape[1] + F.shape[1] - 1
	my_result = vector_to_matrix(convolution_matrix @ vectorI, (output_rows, output_cols))

	print('my result: \n', my_result)
	    
	from scipy import signal
	lib_result = signal.convolve2d(I, F, "full")
	print('lib result: \n', lib_result)

	if (my_result.all() == lib_result.all()):
		print("yay")

	lib_result = signal.convolve2d(I, F, "same")
	print('lib result: \n', lib_result)

	leftpad = (output_rows - I.shape[0])//2
	rightpad = my_result.shape[0]-leftpad-1
	toppad = (output_cols - I.shape[1])//2
	botpad = my_result.shape[1]-toppad-1
	my_result = my_result[leftpad:rightpad, toppad:botpad]
	print('my result: \n', my_result)

#test()