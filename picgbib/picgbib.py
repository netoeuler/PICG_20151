#http://dmr.ath.cx/gfx/python/

import numpy
import matplotlib as mp
import matplotlib.pyplot as plt
import Image

#Salva a imagem carregada em imread
savedImage=None

def imread(image):
	global savedImage
	savedImage = numpy.asarray(Image.open("samples/"+image))
	return savedImage	

def imshow(image):
	if (image.shape[0] < 50):
		plt.imshow(savedImage, interpolation='nearest')
		print 'nearest'
	elif (len(image.shape) == 2): #grayscale
		plt.imshow(grayImage, cmap = plt.get_cmap('gray'))
		print 'grayscale'
	else:
		plt.imshow(savedImage)
		print 'normal'

	plt.show()
	return

def nchannels():
	return savedImage.shape[2]

def size():
	return [savedImage.shape[1], savedImage.shape[0]]

def rgb2gray2():
	grayImage = numpy.dot(savedImage[...,:3], [0.299, 0.587, 0.144])
	return grayImage

def rgb2gray():	
	pesos = [0.299, 0.578, 0.144]		
	divisorRGB=0

	for i in pesos:
		divisorRGB = divisorRGB + i

	
	r = savedImage[:,:,0]
	g = savedImage[:,:,1]
	b = savedImage[:,:,2]

	grayImage = (r*pesos[0] + g*pesos[1] + b*pesos[2]) / divisorRGB	
	'''
	grayImage = []
	for vet3 in savedImage:
		grayVet2 = []
		for vet2 in vet3:
			
			index=0
			dividendoRGB=0			
			for val in vet2:
				if (index==3):
					break

				dividendoRGB = val*pesos[index] + dividendoRGB
				index = index+1

			#print ("antes: "+str(vet2)+" | depois: "+str(int(dividendoRGB / divisorRGB)))
			grayVet2.append( int(dividendoRGB / divisorRGB) )

		grayImage.append(grayVet2)
	'''

	Image.fromarray(grayImage).save("gray_test.tif")	
	
	return grayImage

#def imreadygray(image):

imread("test20.png")
imshow(savedImage)
#rgb2gray2()