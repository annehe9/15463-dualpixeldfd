import skimage
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
import math
import cv2
import scipy
from skimage import img_as_ubyte
import glob
from skimage.transform import rescale
from skimage.color import rgb2gray
from tqdm import tqdm
from skimage.draw import disk
from convmtx2d import convmtx2d, matrix_to_vector, vector_to_matrix
from qpsolvers import solve_qp
from scipy import signal
from homography import homography

print("loading images...")
path = "../data/calib/desktop/more/"
combined_calib = rgb2gray(io.imread(path+"combined.png"))
left_calib = rgb2gray(io.imread(path+"leftcalib.png"))
right_calib = rgb2gray(io.imread(path+"rightcalib.png"))

def find_psf(calib):
	immax = np.max(calib)
	immin = np.min(calib)
	# linearization
	print("linearizing image...")
	calib = (calib-immin)/(immax-immin)
	np.clip(calib, 0, 1, out=calib)

	#calib = homography(calib)

	plt.imshow(calib)
	print("Select upper left and lower right corners.")
	click = plt.ginput(2, -1, show_clicks=True)
	r1 = int(click[0][1])
	r2 = int(click[1][1])
	c1 = int(click[0][0])
	c2 = int(click[1][0])
	calib = calib[r1:r2, c1:c2]

	io.imshow(calib)
	io.show()

	plt.imshow(calib)
	print("Select top, bottom, left, right points of 20 disks in a 5x4(w*h) grid in the center.")
	click = plt.ginput(80, -1, show_clicks=True)
	disks = []
	centers = []
	dim = 0

	print("finding centers...")
	for i in range(0, len(click), 4):
		r1 = int(click[i][1])
		r2 = int(click[i+1][1])
		c1 = int(click[i+2][0])
		c2 = int(click[i+3][0])
		centers.append(((r1+r2+1)//2, (c1+c2+1)//2))
		dim = max(dim, max(r2-r1, c2-c1))
	dim = int(((dim*1.5)//2) * 2) #make it even so halfdim is an integer, also make it slightly bigger
	halfdim = int(dim/2)
	xs = [int(c[0]) for c in centers] # row spacing
	ys = [int(c[1]) for c in centers] # col spacing
	xs = np.asarray(xs).reshape((4,5))
	ys = np.asarray(ys).reshape((4,5))
	xdist = np.mean(np.diff(xs, axis=0))
	ydist = np.mean(np.diff(ys, axis=1))
	spacing = (xdist+ydist)/2
	radius = math.ceil(spacing/5) #radius of sharp disk
	xs = xs.flatten()
	ys = ys.flatten()
	print("slicing images...")
	for i in range(len(centers)):
		disks.append(calib[xs[i]-halfdim:xs[i]+halfdim, ys[i]-halfdim:ys[i]+halfdim])
	G = np.mean(np.asarray(disks), axis=0)
	print(G.shape)
	io.imsave(path+"G.png", G)
	io.imshow(G)
	io.show()

	sharpdisk = np.zeros((dim*4,dim*4), dtype=np.float64)
	rr,cc = disk((halfdim*4,halfdim*4), radius*4)
	sharpdisk[rr,cc] = 1
	#rr, cc, val = skimage.draw.circle_perimeter_aa(halfdim, halfdim, radius)
	#sharpdisk[rr,cc] = val
	sharpdisk = rescale(sharpdisk, 0.25, anti_aliasing = True)
	print(sharpdisk.shape)
	io.imsave(path+"F.png", sharpdisk)
	io.imshow(sharpdisk)
	io.show()

# based off of http://cim.mcgill.ca/~fmannan/CRV16/Calib.pdf and https://github.com/fmannan/PSFEstimation
def find_psf2(G,F):
	x0 =  np.ones(G.shape)
	x0 = x0/np.sum(x0)
	r,c = x0.shape
	A = convmtx2d(F, x0.shape)
	
	# [rr, cc] = np.meshgrid(np.arange(-(r)/2,(r)/2), np.arange(-(c)/2,(c)/2))
	# T = rr**2 + c**2
	# Tn = T/np.max(T)
	# tmp = 1000000 * np.ones(Tn.shape)
	# tmp[2:tmp.shape[0]-1, 2:tmp.shape[1]-1] = Tn[2:Tn.shape[0]-1, 2:Tn.shape[1]-1]
	# Tn = tmp

	H = A.T @ A + np.identity(x0.size)#diag(LAMBDAS(2) * sparse(ones(1, numel(X0))));

	Dx = np.array([-1, 1]).reshape((1,2)) #reshape to make it not 1D
	Dy = np.array([[-1],[1]]).reshape((2,1))
	ADx = convmtx2d(Dx, x0.shape)
	ADy = convmtx2d(Dy, x0.shape)
	H = H + ((ADx.T @ ADx) + (ADy.T @ ADy))
	f = -A @ matrix_to_vector(G)
	print(f.shape)
	output_rows = 2 * G.shape[0] - 1
	output_cols = 2 * G.shape[1] - 1
	f = vector_to_matrix(f, (output_rows, output_cols))
	leftpad = (output_rows - G.shape[0])//2
	rightpad = output_rows-leftpad-1
	toppad = (output_cols - G.shape[1])//2
	botpad = output_cols-toppad-1
	f = f[leftpad:rightpad, toppad:botpad]

	#to find l1 norm use row matrix of 1s of size k
	#l1 = np.ones(x0.size)
	#print(l1.shape)
	#f = f+l1
	
	f = f+1 # i don't think this actually affected the result unfortunately
	f = f.flatten()
	#f = f+Tn.flatten()

	lb = np.zeros(x0.size)
	print(lb.shape)
	x = solve_qp(H,f, None, None, None, None, lb, None)
	print(x)
	return x

#find_psf(combined_calib)
#find_psf(left_calib)
#find_psf(right_calib)
#find_psf(rgb2gray(io.imread("../data/calib/more/left.png")))
#find_psf(rgb2gray(io.imread("../data/calib/more/right.png")))

G_combined = io.imread(path+"Gcombined.png")
F_combined = io.imread(path+"Fcombined.png")
G_left = io.imread(path+"Gleft.png")
F_left = io.imread(path+"Fleft.png")
G_right = io.imread(path+"Gright.png")
F_right = io.imread(path+"Fright.png")
Kcombined = find_psf2(G_combined, F_combined).reshape(G_combined.shape)
Kleft = find_psf2(G_left, F_left).reshape(G_left.shape)
Kright = find_psf2(G_right, F_right).reshape(G_right.shape)

io.imsave(path+"combined_PSF.png", Kcombined)
io.imshow(Kcombined, cmap='gray')
io.show()

io.imsave(path+"left_PSF.png", Kleft)
io.imshow(Kleft, cmap='gray')
io.show()

io.imsave(path+"right_PSF.png", Kright)
io.imshow(Kright, cmap='gray')
io.show()

print(Kcombined.shape)
print(Kleft.shape)
print(Kright.shape)

Kleft = io.imread(path+"left_PSF.png")
Kright = io.imread(path+"right_PSF.png")
#normalized cross correlation
Krflipped = np.fliplr(Kright)

res = cv2.matchTemplate(Krflipped, Kleft, cv2.TM_CCORR_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
print(max_val)

# gdenom = np.var(Kleft)
# gbar = signal.convolve2d(Kleft, np.ones(Kleft.shape)/Kleft.size, boundary="symm", mode="same") #/ np.var(template)
# Kleft = Kleft - gbar
# Krflipped = (Krflipped - signal.convolve2d(Krflipped, np.ones(Kleft.shape)/Kleft.size, boundary="symm", mode="same"))
# #io.imshow(image)
# #io.show()
# corr = signal.correlate2d(Krflipped, Kleft, boundary='symm', mode='same')
# denom = gdenom * signal.convolve2d(Krflipped**2, np.ones(Kleft.shape), boundary = "symm", mode="same")
# corr = corr/(np.sqrt(denom))
# print(corr)
# print(np.max(corr))
#np.save('L.npy', L)
#print("loading L from file...")
#L = np.load('L.npy')
#print(L.shape)