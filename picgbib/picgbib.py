#http://dmr.ath.cx/gfx/python/

import numpy
import matplotlib as mp
import matplotlib.pyplot as plt
import Image
from optparse import OptionParser

def imread(nomeArquivo):
	return numpy.asarray(Image.open("samples/"+nomeArquivo))

def imshow(image):
	if (len(image.shape) == 2): #grayscale
		plt.imshow(image, cmap = plt.get_cmap('gray'))
		if debug:
			print 'grayscale'
	elif (image.shape[0] < 50):
		plt.imshow(image, interpolation='nearest')
		if debug:
			print 'nearest'	
	else:
		plt.imshow(image)
		if debug:
			print 'normal'

	plt.show()
	return

def nchannels(image):
	return image.shape[2]

def size(image):
	return [image.shape[1], image.shape[0]]

def rgb2gray(image):
	grayImage = numpy.dot(image[...,:3], [0.299, 0.587, 0.144])
	if debug:
			print "Gray == original:",grayImage == image #Verifica se a imagem original permanece inalterada
	return grayImage

def imreadgray(nomeArquivo):
	image = numpy.asarray(Image.open("samples/"+nomeArquivo))
	if (len(image.shape) == 2): #grayscale
		return image
	else:
		return rgb2gray(image)

#>>> LINHA DE COMANDO
if __name__ == "__main__":
	parser = OptionParser()
	parser.add_option("-d", "--debug", action="store_true", default=False)
	options, args = parser.parse_args()
	global debug
	debug = options.debug

	img = imread("test20.png")
	#img = imreadgray("lena_std.tif")
	imshow(img)

