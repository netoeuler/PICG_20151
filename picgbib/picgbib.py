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

	if isgrayscale(image):
		plt.imshow(image, cmap = plt.get_cmap('gray'))
		if debug:
			print 'grayscale'	
	else:
		plt.imshow(image)
		if debug:
			print 'normal'
	
	plt.show()
	return

def nchannels(image):
	if len(image.shape) == 2:
		return 1
	else:
		return image.shape[2]

def isgrayscale(image):
	return nchannels(image) == 1

def size(image):
	return [image.shape[1], image.shape[0]]

def rgb2gray(image):
	return numpy.dot(image[...,:3], [0.299, 0.587, 0.144]).astype(numpy.uint8)

def imreadgray(nomeArquivo):
	image = numpy.asarray(Image.open("samples/"+nomeArquivo))
	if isgrayscale(image):
		return image
	else:
		return rgb2gray(image)

def thresh(image, limiar):
	return (image > limiar).astype(numpy.uint8) * 255

def imneg(image):
	return 255 - image

#http://matplotlib.org/1.3.0/examples/pylab_examples/histogram_demo_extended.html
#http://matplotlib.org/examples/statistics/histogram_demo_multihist.html
def hist(image, bin):	
	if nchannels(image) == 1:
		return plt.hist(image.flatten(), bin, color='gray', histtype='bar')[0]
	elif nchannels(image) == 3:
		r=image[...,0].flatten()
		g=image[...,1].flatten()
		b=image[...,2].flatten()
		return plt.hist([r,g,b], bin, histtype='bar', color=['r','g','b'])
		
def showhist (hist):
	#plt.legend()
	return plt.show()

#>>> LINHA DE COMANDO
img = imread("lena_std.tif")
img2 = thresh(img,150)
imshow(img2)
#img2 = hist(img,256)
#showhist(img2)