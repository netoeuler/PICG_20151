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

#Test if g(0,0) is the average of input image
def dft(image):
	m = len(image)
	n = len(image[0])

	#Lines
	x = []
	for i in range(m):
		aux = []
		for j in range(n):
			aux.append(j+0.0)
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
			column.append(part)
		output.append(column)

	#print(output[0][0])

	# divisor = m*n
	
	# soma = 0
	# for a in range(m):
	# 	for b in range(n):
	# 		z.append(b)
	# 		soma += z[a][b]
	
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
	lines = []
	for i in range(size):
		columns = []
		for j in range(size):
			columns.append(color)
		lines.append(columns)
	return numpy.asarray(lines)

def drawLine(image,p0, p1, color):
	if (nchannels(image) != len(color)):
		raise Exception('Nchannels != len(color)')

	dx = p1[0] - p0[0]
	dy = p1[1] - p0[1]

	if (abs(dx) > abs(dy)):
		steps = abs(dx)+1
	else:
		steps = abs(dy)+1

	xinc = dx/float(steps)
	yinc = dy/float(steps)

	x=p0[0]
	y=p0[1]
	
	color2 = numpy.asarray(color)
	image[x,y] = color2
	result = []
	for i in range(steps):
		x += xinc
		y += yinc
		image[x,y] = color2
		result.append([x,y])
	
	return numpy.asarray(result)

def drawCircle(image,c,r,color):
	if (nchannels(image) != len(color)):
		raise Exception('Nchannels != len(color)')

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

	return numpy.asarray(image)

#Use it to test rotate and rotate2
def drawPoly(image, points, color):
	lenpoints = len(points)
	result = []
	for i in range(lenpoints):
		result.append( drawLine(image, points[i%lenpoints], points[(i+1)%lenpoints],color) )

	return numpy.asarray(result)

def rotate(theta, points):
	matrix_rotation = numpy.matrix([[numpy.cos(theta), numpy.sin(theta)], [-numpy.sin(theta), numpy.cos(theta)]])
	xpoint = numpy.array(points)
	result = numpy.dot(xpoint, matrix_rotation)
	return numpy.asarray(result)

def translate(delta, points):
	for i in range(len(points)):
		points[i][0] += delta[0]
		points[i][1] += delta[1]

	return numpy.asarray(points)

def rotate2(theta, delta, points):
	delta_neg = []
	for i in delta:
		delta_neg.append(-1*i)
	
	result = translate(delta_neg, points)
	result = rotate(theta+90, result)
	result = translate(delta, result)

	return numpy.asarray(result)

def ImgRotate(image, alpha):
	img = image



#>>> LINHA DE COMANDO
img = imread("exercicio_fft/lena1.png")
#print(img)
img2 = dft(img)
imshow(img2)

#img = newImage(15,[50,50,50])
#drawCircle(img,[6.0,6.0],3.0,[0.70,0.10,0.50])


#TypeError: Invalid dimensions for image data
#imshow(blur(img))

#rotate2(30,[2,2],[[4,4]])

