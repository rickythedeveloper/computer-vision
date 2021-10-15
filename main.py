from PIL import Image
import numpy as np
import convolution as conv

def main():
	image = Image.open('hiro.png')
	pixels = image.load()
	print('image loaded')
	size = image.size
	r, g, b, a = np.empty(size), np.empty(size), np.empty(size), np.empty(size)
	for i in range(size[0]):
		for j in range(size[1]):
			r[i, j] = pixels[i, j][0]
			g[i, j] = pixels[i, j][1]
			b[i, j] = pixels[i, j][2]
			a[i, j] = pixels[i, j][3]
	print('image data loaded to arrays')
	r_blur = conv.conv_2D_gaussian(r, 20, 51)
	print('Blurred R')
	g_blur = conv.conv_2D_gaussian(g, 20, 51)
	print('Blurred G')
	b_blur = conv.conv_2D_gaussian(b, 20, 51)
	print('Blurred B')
	a_blur = conv.conv_2D_gaussian(a, 20, 51)
	print('Blurred A')
	pass

if __name__ == '__main__':
	main()