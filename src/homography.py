import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from skimage import transform
from skimage import io

def homography(img):
	# basic helper to apply a homography to a mostly rectangular image
	#img = io.imread("")
	plt.imshow(img)
	print("Select screen corners, starting from upper left and going clockwise.")
	click = plt.ginput(4, -1, show_clicks=True)
	r1 = int(click[0][1])
	r2 = int(click[1][1])
	r3 = int(click[2][1])
	r4 = int(click[3][1])
	c1 = int(click[0][0])
	c2 = int(click[1][0])
	c3 = int(click[2][0])
	c4 = int(click[3][0])

	frompts = [[r1,c1],[r2,c2],[r3,c3],[r4,c4]]
	left = min(c1,c3)
	right = max(c2,c4)
	vertical = (right-left)*9/16 # screen ratio
	up = min(r1,r2)
	#down = min(r3,r4)
	down = up+vertical
	topts = [[up,left],[up,right],[down,right],[down,left]]

	frompts = np.array(frompts)
	topts = np.array(topts)
	t = transform.estimate_transform('projective', frompts, topts)
	result = transform.warp(img, t.inverse, mode = 'symmetric')
	#io.imsave("transformed.png", result)
	io.imshow(result)
	io.show()
	return result