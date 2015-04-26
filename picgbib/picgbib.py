import numpy
import matplotlib as mp
import matplotlib.pyplot as plt
import Image

debug=False

def imread(nomeArquivo):
	return numpy.asarray(Image.open("samples/"+nomeArquivo))

def imshow(image):
	if (image.shape[0] < 50):
		plt.imshow(image, interpolation='nearest')
		if debug:
			print 'nearest'	

	if isGrayscale(image):
		plt.imshow(image, cmap = plt.get_cmap('gray'))
		if debug:
			print 'grayscale'	
	else:
		plt.imshow(image)
		if debug:
			print 'normal'
	
	plt.show()
	return

def isGrayscale(image):
	return len(image.shape) == 2

def nchannels(image):
	if isGrayscale(image):
		return 1
	else:
		return image.shape[2]

def size(image):
	return [image.shape[1], image.shape[0]]

def rgb2gray(image):
	return numpy.dot(image[...,:3], [0.299, 0.587, 0.144]).astype(numpy.uint8)

def imreadgray(nomeArquivo):
	image = numpy.asarray(Image.open("samples/"+nomeArquivo))
	if isGrayscale(image):
		return image
	else:
		return rgb2gray(image)

def thresh(image, limiar):
	if isGrayscale(image):
		return image[:,:] > limiar
	else:
		return image[:,:,0] > limiar

def imneg(image):
	if isGrayscale(image):
		return 255 - image[:,:]
	else:
		return 255 - image[:,:,0]

#>>> LINHA DE COMANDO
img = imread("test.jpg")
#img2 = thresh(img,150)
img2 = imneg(img)
imshow(img2)