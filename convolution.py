import numpy as np

def conv_1D_equal_length(a, b):
	if (len(a) != len(b)):
		raise Exception('The lists in 1D convolution must have equal lengths')

	result = 0
	for n in range(len(a)):
		result += a[n] * b[-n-1]
	return result

def conv_1D(a, b):
	if (len(a) == len(b)):
		return [conv_1D_equal_length(a, b)]
	
	length_a, length_b = len(a), len(b)
	if length_a < length_b:
		list_short = a
		list_long = b
		length_short = length_a
		length_long = length_b
	else:
		list_short = b
		list_long = a
		length_short = length_b
		length_long = length_a

	result_length = length_long - length_short + 1
	result = np.empty((result_length,))
	for start_index_long in range(result_length):
		end_index_long = start_index_long + length_short
		result[start_index_long] = conv_1D_equal_length(list_short, list_long[start_index_long:end_index_long])

	return result

def conv_1D2D(list_1d, list_2d, direction=0):
	if direction == 0: # x direction
		result_shape = (
			list_2d.shape[0],
			list_2d.shape[1]-len(list_1d)+1
		)
		result = np.empty(result_shape)
		for j in range(result_shape[0]):
			list_2d_row = list_2d[j]
			row = conv_1D(list_1d, list_2d_row)
			result[j] = row
		return result
	else: # y direction
		return conv_1D2D(list_1d, list_2d.T, 0).T

def conv_2D_gaussian(list_2d, sigma, kernel_size):
	kernel_1d = gaussian_kernel_1d(sigma, kernel_size)
	result1 = conv_1D2D(kernel_1d, list_2d, 0)
	rows = conv_1D2D(kernel_1d, result1, 1)
	return rows


def gaussian_kernel_1d(sigma, size):
	if (size % 2 == 0):
		raise Exception('The kernel size should be odd')
	kernel_center_index = (size - 1) / 2
	kernel = np.zeros(size)
	for index in range(size):
		x = index - kernel_center_index
		kernel[index] = gaussian_1d(x, sigma)
	return kernel

def gaussian_1d(x, sigma):
	return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(- x**2 / (2 * sigma**2))


