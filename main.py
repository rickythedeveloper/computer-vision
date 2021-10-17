from PIL import Image
import numpy as np
import convolution as conv

def getImageFromPixels(r, g, b, a):
	shape = r.shape
	pixels = np.empty(shape, dtype=(np.uint8, 4))
	for j in range(shape[0]):
		for i in range(shape[1]):
			if a is None:
				pixels[j, i] = (r[j, i], g[j, i], b[j, i], 255)
			else:
				pixels[j, i] = (r[j, i], g[j, i], b[j, i], a[j, i])

	image = Image.fromarray(pixels, 'RGBA')
	return image

def main():
	image = Image.open('beach_200.jpg')
	pixels = image.load()
	print('image loaded')
	size = (image.size[1], image.size[0]) # (height, width)
	pixel_data_length = len(pixels[0,0])
	r, g, b = np.empty(size), np.empty(size), np.empty(size)
	if pixel_data_length == 4:
		a = np.empty(size)
	for j in range(size[0]):
		for i in range(size[1]):
			r[j, i] = pixels[i, j][0]
			g[j, i] = pixels[i, j][1]
			b[j, i] = pixels[i, j][2]

			if pixel_data_length == 4:
				a[j, i] = pixels[i, j][3]

	sigma, kernel_size = 3, 23
	print('image data loaded to arrays')
	r_blur = conv.conv_2D_gaussian(r, sigma, kernel_size)
	print('Blurred R')
	g_blur = conv.conv_2D_gaussian(g, sigma, kernel_size)
	print('Blurred G')
	b_blur = conv.conv_2D_gaussian(b, sigma, kernel_size)
	print('Blurred B')
	if pixel_data_length == 4:
		a_blur = conv.conv_2D_gaussian(a, sigma, kernel_size)
		print('Blurred A')
	else:
		a_blur = None

	image = getImageFromPixels(r_blur, g_blur, b_blur, a_blur)
	image.show()
	pass

if __name__ == '__main__':
	main()