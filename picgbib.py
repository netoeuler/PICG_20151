import numpy
import matplotlib as mp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import exp
#import Image
import math
from scipy import signal as sg

debug=False

def imread(nomeArquivo):
	#return numpy.asarray(Image.open(nomeArquivo))
	return mpimg.imread(nomeArquivo)

def imshow(image):
	if isgrayscale(image):
		plt.imshow(image, cmap = plt.get_cmap('gray'), interpolation='nearest')
		if debug:
			print('grayscale')
	else:
		plt.imshow(image, interpolation='nearest')
		if debug:
			print('normal')
	
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
		g[i] = (r*(image[i]-m))+m

	return numpy.asarray(g)

#http://juanreyero.com/article/python/python-convolution.html
def convolve(image, mask):
	return sg.convolve(image, mask, "valid")

def maskBlur():
	return numpy.asarray([0.0625*(numpy.asarray([[1,2,1],[2,4,2],[1,2,1]]))])

def blur(image):
	return convolve(image,maskBlur())

def seSquare3():
	return numpy.asarray([[1,1,1],[1,1,1],[1,1,1]])

def seCross3():
	return numpy.asarray([[0,1,0],[1,1,1],[0,1,0]])

def erode(image, se):
	smaller=-1
	return

def dilate(image, se):
	bigger=-1
	return 

#Testar se g(0,0) é a média da imagem de entrada
def dft(image):
	m = len(image)
	n = len(image[0])

	#Lines
	x = []
	for i in range(m):
		aux = []
		for j in range(n):
			aux.append(j + 0.0)
		x.append(aux)

	#Columns
	y = []
	for i in range(n):
		aux = [i+0.0] * m
		y.append(aux)

	output = []

	for u in range(m):
		column = []
		for v in range(n):			
			angle = float(2 * math.pi * (u*x[u][v])/m + (v*y[u][v])/n)
			part = 1/(m*n) * math.cos(angle) - 1j*math.sin(angle)
			column.append(part.real)
		output.append(column)

	print(output[0][0])


	divisor = m*n
	
	soma = 0
	for a in range(m):
		for b in range(n):
			z.append(b)
			soma += z[a][b]
	
	test = soma / divisor

	return numpy.asarray(output)

def idft(image):
	m = len(image)
	n = len(image[0])

	#Lines
	x = []
	for i in range(m):
		aux = []
		for j in range(n):
			aux.append(j + 0.0)
		x.append(aux)

	#Columns
	y = []
	for i in range(n):
		aux = [i+0.0] * m
		y.append(aux)

	output = []

	for u in range(m):
		column = []
		for v in range(n):			
			angle = float(2 * math.pi * (u*x[u][v])/m + (v*y[u][v])/n)
			part = math.cos(angle) + 1j*math.sin(angle)			
			column.append(part.real)
		output.append(column)
	
	return numpy.asarray(output)

def newImage(size,color):
	# x=[]
	# y=[]
	# for i in size:
	# 	x.append(i[0])
	# for i in size:
	# 	y.append(i[1])
	
	plt.plot(x,y,color=color)

def drawLine(image,p0, p1, color):
	if (nchannels(image) != len(color)):
		print('Nchannels != len(color)')		
		return

	dx = p1[0] - p0[0]
	dy = p1[1] - p0[1]

	if (dx > dy):
		steps = abs(dx)
	else:
		steps = abs(dy)

	xinc = dx/float(steps)
	yinc = dy/float(steps)

	x=0
	y=0
	
	#points = []	
	color2 = numpy.asarray(color)
	for i in range(steps+1):
		x += xinc
		y += yinc
		#points.append([x,y])		
		image[x,y] = color2
	
	#newImage(points,tuple(color))
	imshow(image)

def drawCircle(image,c,r,color):
	if (nchannels(image) != len(color)):
		print('Nchannels != len(color)')
		return

	x=0.0
	y=r
	xc=c[0]
	yc=c[1]

	color2 = numpy.asarray(color)
	image[round(xc+x),round(yc+y)] = color2
	image[round(xc+x),round(yc-y)] = color2
	image[round(xc+y),round(yc+x)] = color2
	image[round(xc-yc),round(yc+x)] = color2
	
	while x <= y:
		x = x + 1.0
		y = math.sqrt((r*r) - (x*x))
		image[round(xc+x),round(yc+y)] = color2
		image[round(xc+x),round(yc-y)] = color2
		image[round(xc-x),round(yc+x)] = color2
		image[round(xc-x),round(yc-y)] = color2
		image[round(xc+y),round(yc+x)] = color2
		image[round(xc+y),round(yc-x)] = color2
		image[round(xc-y),round(yc+x)] = color2
		image[round(xc-y),round(yc-x)] = color2

	#points = [[round(xc+x),round(xc+x),round(xc+y),round(xc-yc)],[round(yc+y),round(yc-y),round(yc+x),round(yc+x)]]

	#newImage(points,tuple(color))
	imshow(image)


#>>> LINHA DE COMANDO
img = imread("exercicio_fft/lena1.jpg")
#imshow(convolve(img,[[[1, -1]]]))
#img2 = dft(img)

#drawLine(img,[1,0],[5,2],[0.70,0.10,0.50])
drawCircle(img,[6.0,6.0],3.0,[0.70,0.10,0.50])

#TypeError: Invalid dimensions for image data
#imshow(blur(img))