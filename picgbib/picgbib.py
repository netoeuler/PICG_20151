#http://dmr.ath.cx/gfx/python/

import numpy
import matplotlib as mp
import matplotlib.pyplot as plt
import Image

def imread(nomeArquivo):
	savedImage = numpy.asarray(Image.open("samples/"+nomeArquivo))
	return savedImage	

def imshow(image):
	if (image.shape[0] < 50):
		plt.imshow(image, interpolation='nearest')
		print 'nearest'
	elif (len(image.shape) == 2): #grayscale
		plt.imshow(image, cmap = plt.get_cmap('gray'))
		print 'grayscale'
	else:
		plt.imshow(image)
		print 'normal'

	plt.show()
	return

def nchannels(image):
	return image.shape[2]

def size(image):
	return [image.shape[1], image.shape[0]]

def rgb2gray(image):
	grayImage = numpy.dot(image[...,:3], [0.299, 0.587, 0.144])
	print grayImage == image #Verifica se a imagem original permanece inalterada
	return grayImage

#def imreadygray(nomeArquivo):

#>>> LINHA DE COMANDO

img = imread("test20.png")
rgb2gray(img)
#imshow(img)
