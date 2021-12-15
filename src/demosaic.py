import skimage
from skimage import io
import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate
from skimage import img_as_ubyte
import cv2
from skimage.transform import rescale

# this was adapted from my 15463 assignment 1

img_arr = np.fromfile("../data/more/outside.raw", dtype='>u2')

shape = img_arr.shape
print(shape)
img_arr = np.reshape(img_arr, (3024,4032))
shape = img_arr.shape
dt = img_arr.dtype
print(dt.itemsize)
print(dt.name)

# cv2.imwrite('12345a.jpg', np.uint8(cv2.cvtColor(img_arr, cv2.COLOR_BayerBG2BGR) / 256))
# cv2.imwrite('12345b.jpg', np.uint8(cv2.cvtColor(img_arr, cv2.COLOR_BayerRG2BGR) / 256))
# cv2.imwrite('12345c.jpg', np.uint8(cv2.cvtColor(img_arr, cv2.COLOR_BayerGB2BGR) / 256))
# cv2.imwrite('12345d.jpg', np.uint8(cv2.cvtColor(img_arr, cv2.COLOR_BayerGR2BGR) / 256))

#print("screaming")
img_arr = img_arr.astype('float64')

# linearization
print("linearizing image...")
img_arr = (img_arr - 1000)/ (2**14-1-1000)
np.clip(img_arr, 0, 1, out=img_arr)
io.imshow(img_arr)
io.show()

gleft = io.imread("../data/more/Outside_L.pgm")
print(gleft.shape)
gdt = gleft.dtype
print(gdt.itemsize)
print(gdt.name)
print(np.max(gleft))
gleft = (gleft-1000)/(2**14-1000)
np.clip(gleft, 0, 1, out=gleft)
gright = io.imread("../data/more/Outside_R.pgm")
gright = (gright-1000)/(2**14-1000)
np.clip(gright, 0, 1, out=gright)
io.imshow(gleft)
io.show()

#gx = np.arange(0, gleft.shape[1])
#gy = np.arange(0, gleft.shape[0])
#fgl = interpolate.interp2d(gx,gy,gleft)
#fgr = interpolate.interp2d(gx,gy,gright)
#gx = np.arange(0, gleft.shape[1])
#gy = np.arange(0, gleft.shape[0]*2)
gleft = rescale(gleft, (2, 1), anti_aliasing = False)
gright = rescale(gright, (2, 1), anti_aliasing = False)
#gleft = fgl(gx,gy)
#gright = fgr(gx,gy)
io.imshow(gleft)
io.show()

# white balancing
def gray_world(im):
	print("white balancing using gray world...")
	red = im[1::2, 1::2]
	green1 = im[0::2, 1::2]
	green2 = im[1::2, 0::2]
	blue = im[0::2, 0::2]
	ravg = np.mean(red)
	gavg = (np.mean(green1) + np.mean(green2))/2
	bavg =  np.mean(blue)
	print(gavg/ravg)
	print(gavg/bavg)
	im[1::2, 1::2] = [x*(gavg/ravg) for x in im[1::2, 1::2]]
	im[0::2, 0::2] = [x*(gavg/bavg) for x in im[0::2, 0::2]]
	print("done")
	print("showing image")
	io.imshow(im)
	io.show()
	return im

# white balancing
def gray_world_known_green(im, green):
	print("white balancing using gray world...")
	red = im[1::2, 1::2]
	blue = im[0::2, 0::2]
	ravg = np.mean(red)
	gavg = np.mean(green)
	bavg =  np.mean(blue)
	print(gavg/ravg)
	print(gavg/bavg)
	im[1::2, 1::2] = [x*(gavg/ravg) for x in im[1::2, 1::2]]
	im[0::2, 0::2] = [x*(gavg/bavg) for x in im[0::2, 0::2]]
	print("done")
	print("showing image")
	io.imshow(im)
	io.show()
	return im

def white_world(img_arr):
	print("white balancing using white world...")
	red = img_arr[0::2, 1::2]
	green1 = img_arr[0::2, 0::2]
	green2 = img_arr[1::2, 1::2]
	blue = img_arr[1::2, 0::2]
	rmax = np.max(red)
	print("rmax:", rmax)
	gmax = max(np.max(green1), np.max(green2))
	print("gmax1:", np.max(green2))
	bmax =  np.max(blue)
	print("bmax", bmax)
	print(gmax/rmax)
	print(gmax/bmax)
	img_arr[0::2, 1::2] = [x*(gmax/rmax) for x in img_arr[0::2, 0::2]]
	img_arr[1::2, 0::2] = [x*(gmax/bmax) for x in img_arr[1::2, 1::2]]
	print("done")
	print("showing image")
	io.imshow(img_arr)
	io.show()

#whoops I did this post processing
def manual_whitebalancing(img):
	fig = plt.figure()
	fig.add_subplot(1,2,1)
	plt.imshow(img)
	click = plt.ginput()
	print(click)
	r = img[int(click[0][1]), int(click[0][0]), 0]
	g = img[int(click[0][1]), int(click[0][0]), 1]
	b = img[int(click[0][1]), int(click[0][0]), 2]
	print(g/r)
	print(g/b)
	newimg = img * np.array([g/r, 1, g/b])
	fig.add_subplot(1,2,2)
	plt.imshow(newimg)
	plt.savefig('output.png')
	plt.show()

#this is the one for pre processing
def manual_whitebalancing1(img_arr):
	#fig = plt.figure()
	#fig.add_subplot(1,2,1)
	plt.imshow(img_arr)
	click = plt.ginput()
	print(click)
	rbegin = (int(click[0][1]) //2)*2
	cbegin = (int(click[0][0])//2)*2
	r = img_arr[rbegin, cbegin]
	g = (img_arr[rbegin+1 ,cbegin] + img_arr[rbegin, cbegin+1])/2
	b = img_arr[rbegin+1, cbegin+1]
	print(g/r)
	print(g/b)
	img_arr[0::2, 0::2] = [x*(g/r) for x in img_arr[0::2, 0::2]]
	img_arr[1::2, 1::2] = [x*(g/b) for x in img_arr[1::2, 1::2]]
	#fig.add_subplot(1,2,2)
	#plt.imshow(newimg)
	#plt.savefig('output.png')
	#plt.show()

#manual_whitebalancing1(img_arr)
combined = gray_world(img_arr)
#white_world(img_arr)

red = combined[1::2, 1::2]
green1 = combined[0::2, 1::2]
green2 = combined[1::2, 0::2]
blue = combined[0::2, 0::2]
#show
rgb = np.dstack((red,(green1+green2)/2,blue))
np.clip(rgb, 0, 1, out=rgb)
plt.imshow(img_as_ubyte(rgb))
plt.show()

# demosaicing
redx = np.arange(1, shape[1], 2)
redy = np.arange(1, shape[0], 2)
bluex = np.arange(0, shape[1], 2)
bluey = np.arange(0, shape[0], 2)
greenx1 = np.arange(1, shape[1], 2)
greeny1 = np.arange(0, shape[0], 2)
greenx2 = np.arange(0, shape[1], 2)
greeny2 = np.arange(1, shape[0], 2)

fr = interpolate.interp2d(redx, redy, red)
fb = interpolate.interp2d(bluex, bluey, blue)
fg1 = interpolate.interp2d(greenx1, greeny1, green1)
fg2 = interpolate.interp2d(greenx2, greeny2, green2)
x = np.arange(0,shape[1])
y = np.arange(0,shape[0])
resultr = fr(x,y)
resultb = fb(x,y)
resultg1 = fg1(x,y)
resultg2 = fg2(x,y)
rgb = np.dstack((resultr,(resultg1+resultg2)/2,resultb))
plt.imshow(rgb)
plt.show()

m_srgb_xyz = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
m_xyz2cam = np.array([[0.6988,-0.1384,-0.0714],[-0.5631,1.3410,0.2447],[-0.1485,0.2204,0.7318]])

m_srgb2cam = np.matmul(m_xyz2cam, m_srgb_xyz)
rowsum = m_srgb2cam.sum(axis=1)
m_srgb2cam_norm = m_srgb2cam/rowsum[:, np.newaxis]
rgb = rgb @ m_srgb2cam_norm.T #from alice's advice in slack
#plt.imshow(rgb)
#plt.show()

# brightness adjustment
rgb = rgb * 0.9
np.clip(rgb, None, 1, out=rgb)
gray = skimage.color.rgb2gray(rgb)
print(np.mean(gray)) # got 0.0722483346654103 without scaling
#plt.imshow(rgb)
#plt.show()

#gamma encoding
np.where( rgb <= 0.0031308, 12.92*rgb, (1+0.055)*pow(rgb, 1/2.4)-0.055 )
plt.imshow(rgb)
plt.show()

#manual_whitebalancing(rgb)

io.imsave("../data/more/Outside_B.png", img_as_ubyte(rgb))
#io.imsave("campus_jpg95.jpeg", img_as_ubyte(rgb), quality=60)



#left and right images
left = gray_world_known_green(img_arr, gleft)
right = gray_world_known_green(img_arr, gright)

red = left[1::2, 1::2]
blue = left[0::2, 0::2]
rgb_left = np.dstack((red,gleft,blue))
np.clip(rgb_left, 0, 1, out=rgb_left)
#plt.imshow(img_as_ubyte(rgb))
#plt.show()

# demosaicing
redx = np.arange(1, shape[1], 2)
redy = np.arange(1, shape[0], 2)
bluex = np.arange(0, shape[1], 2)
bluey = np.arange(0, shape[0], 2)
greenx1 = np.arange(1, shape[1], 2)
greeny1 = np.arange(0, shape[0], 2)
greenx2 = np.arange(0, shape[1], 2)
greeny2 = np.arange(1, shape[0], 2)
fr = interpolate.interp2d(redx, redy, red)
fb = interpolate.interp2d(bluex, bluey, blue)
fg1 = interpolate.interp2d(greenx1, greeny1, gleft)
fg2 = interpolate.interp2d(greenx2, greeny2, gleft)
x = np.arange(0,shape[1])
y = np.arange(0,shape[0])
resultr = fr(x,y)
resultb = fb(x,y)
resultg1 = fg1(x,y)
resultg2 = fg2(x,y)
rgb_left = np.dstack((resultr,(resultg1+resultg2)/2,resultb))
plt.imshow(rgb_left)
plt.show()

red = right[1::2, 1::2]
blue = right[0::2, 0::2]
fr = interpolate.interp2d(redx, redy, red)
fb = interpolate.interp2d(bluex, bluey, blue)
fg1 = interpolate.interp2d(greenx1, greeny1, gright)
fg2 = interpolate.interp2d(greenx2, greeny2, gright)
x = np.arange(0,shape[1])
y = np.arange(0,shape[0])
resultr = fr(x,y)
resultb = fb(x,y)
resultg1 = fg1(x,y)
resultg2 = fg2(x,y)
rgb_right = np.dstack((resultr,(resultg1+resultg2)/2,resultb))
plt.imshow(rgb_right)
plt.show()

m_srgb_xyz = np.array([[0.4124564, 0.3575761, 0.1804375], [0.2126729, 0.7151522, 0.0721750], [0.0193339, 0.1191920, 0.9503041]])
m_xyz2cam = np.array([[0.6988,-0.1384,-0.0714],[-0.5631,1.3410,0.2447],[-0.1485,0.2204,0.7318]])

m_srgb2cam = np.matmul(m_xyz2cam, m_srgb_xyz)
rowsum = m_srgb2cam.sum(axis=1)
m_srgb2cam_norm = m_srgb2cam/rowsum[:, np.newaxis]
rgb_left = rgb_left @ m_srgb2cam_norm.T #from alice's advice in slack
rgb_right = rgb_right @ m_srgb2cam_norm.T #from alice's advice in slack
#plt.imshow(rgb)
#plt.show()

# brightness adjustment
rgb_left = rgb_left * 0.9
np.clip(rgb_left, None, 1, out=rgb_left)
gray = skimage.color.rgb2gray(rgb_left)
print(np.mean(gray)) # got 0.0722483346654103 without scaling
rgb_right = rgb_right * 0.9
np.clip(rgb_right, None, 1, out=rgb_right)
gray = skimage.color.rgb2gray(rgb_right)
print(np.mean(gray)) # got 0.0722483346654103 without scaling

#gamma encoding
np.where( rgb_left <= 0.0031308, 12.92*rgb_left, (1+0.055)*pow(rgb_left, 1/2.4)-0.055 )
plt.imshow(rgb_left)
plt.show()
io.imsave("../data/more/Outside_L.png", img_as_ubyte(rgb_left))

np.where( rgb_right <= 0.0031308, 12.92*rgb_right, (1+0.055)*pow(rgb_right, 1/2.4)-0.055 )
plt.imshow(rgb_right)
plt.show()
io.imsave("../data/more/Outside_R.png", img_as_ubyte(rgb_right))