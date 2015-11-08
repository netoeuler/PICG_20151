import numpy
import matplotlib as mp
import matplotlib.pyplot as plt
import Image
import math

debug=False

def imread(nomeArquivo):
	return numpy.asarray(Image.open(nomeArquivo))

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
	image = numpy.asarray(Image.open(nomeArquivo))
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
		
def showhist(hist):
	#plt.legend()
	return plt.show()

#Falta a letra 'a'
def contrast(image,r,m):
	n = len(image)

	g = [0.0] * n

	for i in range(n):
		g[i] = r*image[i]+m

	return numpy.asarray(g)

#under construction
def dft(image):
	n = len(image)

	inreal = image;
	inimag = [complex(0)] * n;

	outreal = [0.0] * n
	outimag = [0.0] * n

	for k in range(n):
		sumreal = 0.0
		sumimag = 0.0
		for t in range(n):
			angle = 2*math.pi*t*k/n
			sumreal +=  inreal[t] * math.cos(angle) + inimag[t] * math.sin(angle)
			sumimag += -inreal[t] * math.sin(angle) + inimag[t] * math.cos(angle)
        outreal[k] = sumreal
        outimag[k] = sumimag

	output = [0.0] * n

	for k in range(n):
		output[k] = outreal[k] + outimag[k]

	return numpy.asarray(output)

#>>> LINHA DE COMANDO
img = imread("exercicio_fft/lena1.jpg")
#imshow(contrast(img,5,2))
imshow(dft(img))
#img2 = hist(img,256)
#showhist(img2)