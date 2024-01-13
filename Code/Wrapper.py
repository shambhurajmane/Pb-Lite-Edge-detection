#!/usr/bin/env python3

"""
RBE/CS549 Spring 2022: Computer Vision
Homework 0: Alohomora: Phase 1 Starter Code

Colab file can be found at:
	https://colab.research.google.com/drive/1FUByhYCYAfpl8J9VxMQ1DcfITpY8qgsF

Author(s): 
Prof. Nitin J. Sanket (nsanket@wpi.edu), Lening Li (lli4@wpi.edu), Gejji, Vaishnavi Vivek (vgejji@wpi.edu)
Robotics Engineering Department,
Worcester Polytechnic Institute

Code adapted from CMSC733 at the University of Maryland, College Park.
"""

# Code starts here:

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import sys
from scipy.ndimage import rotate
import os
import sklearn.cluster

# Set all directory 
filter_directory = "./data/filters"
image_folder_name = "./BSDS500/Images"
op_folder_name = "./data/op_images"



# To create filter images from given numpy array
def create_filter_img(filter_bank, file_name, cols, size):
	rows = int(np.ceil(len(filter_bank)/cols))
	plt.subplots(rows, cols, figsize=size)
	for index in range(len(filter_bank)):
		plt.subplot(rows, cols, index+1)
		plt.axis('off')
		plt.imshow(filter_bank[index], cmap='gray')
	folder_name =  "filters/" 
	if not os.path.exists(folder_name):
		os.makedirs(folder_name)	
	
	file_name =  folder_name +file_name

	plt.savefig(file_name)
	plt.close()
	

# To keep the image format standard

def jpg2pngList(jpg_list):
	png_list = []
	for n in range(len(jpg_list)):
		s = jpg_list[n]
		f_name = str()
		for i in range(len(s)):
			if s[i] == '.':
				break

			f_name += str(s[i])	

		png_list.append(str(f_name) + ".png")
	
	return png_list

# Convolve function mathematical interpretation 
def convolve(given_image,gaussian_kernel ):
	# print(sobel_filter[0,0])
	kenrnel_size = given_image.shape[0]
	image_size_x = given_image.shape[0]
	image_size_y = given_image.shape[1]


	filter = np.zeros((image_size_x,image_size_y))
	shift_x = int(image_size_x/2)
	shift_y = int(image_size_y/2)


	for i in range(image_size_x):
		for j in range(image_size_y):
			val=0
			for m in range(kenrnel_size):
				for n in range(kenrnel_size):
					try:
						# print(i-shift+m)
						val+= given_image[i-shift_x+m, j-shift_y+n] * gaussian_kernel[m][n]
					except IndexError:
						continue
			filter[i][j] = val       
	return filter


def generate_dog_filter_bank(scales, orientations, filter_size):
	print("generating DoG filter")
	dog_filter_bank = []
	sobel_filter = np.matrix([[-1,-2,0,2,1],\
							[-2,-3,0,3,2],\
							[-3,-5,0,5,3],\
							[-2,-3,0,3,2],\
							[-1,-2,0,2,1]])/48
	# sobel_filter = np.matrix([[-1,0,1],\
	# 						[-2,0,2],\
	# 						[-1,0,1]])/8
	shape , _ = sobel_filter.shape
	pad_size= int((filter_size - shape)/2)
	sobel_filter= np.pad(sobel_filter,(pad_size,pad_size),'constant', constant_values=0)
	
	for scale in scales:
		sigma =scale
		kenrnel_size = filter_size
		shift = int(kenrnel_size/2)
		gaussian_kernel = np.zeros((kenrnel_size,kenrnel_size))
		# gaussian_kernel mathematical interpretation 
		for i in range(kenrnel_size):
			for j in range(kenrnel_size):
				i_shift, j_shift= abs(shift- i) , abs(shift- j) 
				gaussian_kernel[i][j] = (1/(2*np.pi *sigma*sigma))* np.exp((-0.5)*((i_shift*i_shift + j_shift*j_shift)/(sigma*sigma)))
		for orientation in orientations:
			rotated_sobel_filter = rotate(sobel_filter, angle=orientation, reshape=False)
			# print(gaussian_kernel)
			dog = convolve(rotated_sobel_filter, gaussian_kernel)      
			dog_filter_bank.append(dog)
	print("dog filter genearted")
	return dog_filter_bank

def create_lm_bank(scales, orientations, filter_size):
	print("generating Leung-Malik Filters")

	lm_filter_bank = []
	lml_scales= scales[0:3]
	log_scale = scales + [i * 3 for i in scales]

	# gradient_filter mathematical interpretation 
	gradient_filter = np.matrix([[-1,-2,0,2,1],\
							[-2,-3,0,3,2],\
							[-3,-5,0,5,3],\
							[-2,-3,0,3,2],\
							[-1,-2,0,2,1]])
	shape , _ = gradient_filter.shape
	g_pad_size= int((filter_size - shape)/2)
	gradient_filter= np.pad(gradient_filter,(g_pad_size,g_pad_size),'constant', constant_values=0)

	# laplacian_filter mathematical interpretation 
	laplacian_filter = np.matrix([[0,1,0],[1,-4,1],[0,1,0]])
	shape , _ = laplacian_filter.shape
	l_pad_size= int((filter_size - shape)/2)
	laplacian_filter= np.pad(laplacian_filter,(l_pad_size,l_pad_size),'constant', constant_values=0)
	print(laplacian_filter.shape, orientations)

	lm_filter_bank = lm_generate_1st_order_gaussian(lml_scales,orientations, gradient_filter,lm_filter_bank,filter_size)
	lm_filter_bank = lm_generate_2nd_order_gaussian(lml_scales,orientations, laplacian_filter,lm_filter_bank,filter_size)
	lm_filter_bank = lm_generate_log(log_scale, laplacian_filter,lm_filter_bank,filter_size)
	lm_filter_bank = lm_generate_gaussions(scales, lm_filter_bank)
	print("Leung-Malik Filters generated")
		
	return lm_filter_bank

def lm_generate_1st_order_gaussian(lml_scales,orientations, gradient_filter,lm_filter_bank,filter_size ):
	print("generating lm_generate_1st_order_gaussian filter")

	for scale in lml_scales:
		kenrnel_size =filter_size
		shift = int(kenrnel_size/2)
		sigma_x =scale
		sigma_y =scale *3
		LM = np.zeros((51,51))
		gaussian_kernel = np.zeros((kenrnel_size,kenrnel_size))
		for i in range(kenrnel_size):
			for j in range(kenrnel_size):
				i_shift, j_shift= abs(shift- i) , abs(shift- j) 
				gaussian_kernel[i][j] = (1/(2*np.pi *sigma_x*sigma_y))* np.exp((-0.5)*((i_shift*i_shift/(sigma_x*sigma_x)) + (j_shift*j_shift/(sigma_y*sigma_y))))
		for orientation in orientations:
			oc=0
			# print(gaussian_kernel)
			rotated_gaussian_kernel = rotate(gaussian_kernel, angle=orientation, reshape=False)

			LM = convolve(gradient_filter, rotated_gaussian_kernel)      
			lm_filter_bank.append(LM)
	print("lm_generate_1st_order_gaussian filter generated")
	return lm_filter_bank

def lm_generate_2nd_order_gaussian(lml_scales,orientations, laplacian_filter,lm_filter_bank,filter_size ):
	print("generating lm_generate_2nd_order_gaussian filter")
	
	for scale in lml_scales:
		sigma_x =scale
		sigma_y =scale *3
		kenrnel_size =filter_size
		# rotated_sobel_filter = rotate(laplacian_filter, angle=orientation, reshape=False)
		shift = int(kenrnel_size/2)
		LM = np.zeros((51,51))
		gaussian_kernel = np.zeros((kenrnel_size,kenrnel_size))
		for i in range(kenrnel_size):
			for j in range(kenrnel_size):
				i_shift, j_shift= abs(shift- i) , abs(shift- j) 
				gaussian_kernel[i][j] = (1/(2*np.pi *sigma_x*sigma_y))* np.exp((-0.5)*((i_shift*i_shift/(sigma_x*sigma_x)) + (j_shift*j_shift/(sigma_y*sigma_y))))
			

		for orientation in orientations:
			# print(gaussian_kernel)
			rotated_gaussian_kernel = rotate(gaussian_kernel, angle=orientation, reshape=False)
			LM = convolve(laplacian_filter, rotated_gaussian_kernel )      
			lm_filter_bank.append(LM)
	print("lm_generate_2nd_order_gaussian filter generated")
	
	return lm_filter_bank

def lm_generate_log(log_scale,laplacian_filter,lm_filter_bank,filter_size ):
	print("generating lm_generate_log filter")

	for scale in log_scale:
		sigma =scale
		kenrnel_size = filter_size
		shift = int(kenrnel_size/2)
		LM = np.zeros((51,51))
		gaussian_kernel = np.zeros((kenrnel_size,kenrnel_size))
		for i in range(kenrnel_size):
			for j in range(kenrnel_size):
				i_shift, j_shift= abs(shift- i) , abs(shift- j) 
				gaussian_kernel[i][j] = (1/(2*np.pi *sigma*sigma))* np.exp((-0.5)*((i_shift*i_shift + j_shift*j_shift)/(sigma*sigma)))
		# print(gaussian_kernel)
		
		# print(sobel_filter[0,0])
		LM = convolve(laplacian_filter, gaussian_kernel)      
      
		lm_filter_bank.append(LM)
	print("lm_generate_log filter generated")
	
	return lm_filter_bank
	
def lm_generate_gaussions(scales,lm_filter_bank ):
	print("generating lm_generate_gaussions filter")

	for scale in scales:
		sigma =scale
		kenrnel_size = 30
		shift = int(kenrnel_size/2)
		gaussian_kernel = np.zeros((kenrnel_size,kenrnel_size))
		for i in range(kenrnel_size):
			for j in range(kenrnel_size):
				i_shift, j_shift= abs(shift- i) , abs(shift- j) 
				gaussian_kernel[i][j] = (1/(2*np.pi *sigma*sigma))* np.exp((-0.5)*((i_shift*i_shift + j_shift*j_shift)/(sigma*sigma)))
		# print(gaussian_kernel)
		    
		lm_filter_bank.append(gaussian_kernel)
	print("lm_generate_gaussions filter generated")
	
	return lm_filter_bank
	
def generate_sinusoidal(frequency, size, angle):
	if (size%2) == 0:
		index = size/2
	else:
		index = (size - 1)/2

	x, y = np.meshgrid(np.linspace(-index, index, size), np.linspace(-index, index, size))
	x_prime = x * np.cos(angle) + y * np.sin(angle)
	generate_sinusoidal = np.sin(x_prime * 2 * np.pi * frequency/size)

	return generate_sinusoidal

def create_gabor_bank(scales, orientations, frequencies):
	print("generating gabor filter")

	gabor_filter_bank = []

	for count in range(len(scales)):
		sigma =scales[count]		
		kenrnel_size = 51
		shift = int(kenrnel_size/2)
		gaussian_kernel = np.zeros((kenrnel_size,kenrnel_size))
		for i in range(kenrnel_size):
			for j in range(kenrnel_size):
				i_shift, j_shift= abs(shift- i) , abs(shift- j) 
				gaussian_kernel[i][j] = (1/(2*np.pi *sigma*sigma))* np.exp((-0.5)*((i_shift*i_shift + j_shift*j_shift)/(sigma*sigma)))
		# print(gaussian_kernel)
		for orientation in orientations:
			sinusoidal = generate_sinusoidal(frequencies[count], kenrnel_size, orientation)
			gabor_filter = gaussian_kernel * sinusoidal
			gabor_filter_bank.append(gabor_filter)
	print("gabor filter generated")
		
	return gabor_filter_bank

def loadImages(folder_name, files):
	print("Loading images from ", folder_name)
	images = []
	if files == None:
		files = os.listdir(folder_name)
	print(files)
	for file in files:
		image_path = folder_name + "/" + file
		image = cv2.imread(image_path)
		if image is not None:
			images.append(image)
			
		else:
			print("Error in loading image ", image)

	return images

def create_HalfDisc(radii,orientations):
	print("generating HalfDisc mask")

	half_disk_bank = []
	for radius in radii:
		half_disk = np.zeros((radius*2+1,radius*2+1))
		center = ((radius*2+1) // 2, (radius*2+1) // 2)
		# Draw a filled white circle on the black image
		cv2.circle(half_disk, center, radius, (255, 255, 255), -1)

		# Create a half disk mask by setting the lower half of the image to black
		half_disk[center[1]:, :] = 0

		for orientation in orientations:
			rotated = cv2.getRotationMatrix2D(center, orientation, 1.0)
			rotated = cv2.warpAffine(half_disk, rotated, (radius*2+1, radius*2+1))
			half_disk_bank.append(rotated)
			rotated_half_disk = rotate(rotated, angle=180)
			half_disk_bank.append(rotated_half_disk)
		

	print("generated HalfDisc mask")

	return half_disk_bank





def chisquareDistance(input, bins, filter_bank):

	chi_square_distances = []
	N = len(filter_bank)
	n = 0
	while n < N:
		left_mask = filter_bank[n]
		right_mask = filter_bank[n+1]		
		tmp = np.zeros(input.shape)
		chi_sq_dist = np.zeros(input.shape)
		min_bin = np.min(input)
	

		for bin in range(bins):
			tmp[input == bin+min_bin] = 1
			g_i = cv2.filter2D(tmp,-1,left_mask)
			h_i = cv2.filter2D(tmp,-1,right_mask)
			chi_sq_dist += (g_i - h_i)**2/(g_i + h_i + np.exp(-7))

		chi_sq_dist /= 2
		chi_square_distances.append(chi_sq_dist)
		n = n+2
    	

	return chi_square_distances


def main():

	"""
	Generate Difference of Gaussian Filter Bank: (DoG)
	Display all the filters in this filter bank and save image as DoG.png,
	use command "cv2.imwrite(...)"
	"""
	scales = [1,3]
	orientations = np.linspace(0,360,num = 16)
	filter_size=25
	dog_filter_bank = generate_dog_filter_bank(scales,orientations,filter_size)
	create_filter_img(dog_filter_bank ,'DoG.png',16 , (16,6))


	"""
	Generate Leung-Malik Filter Bank: (LM)
	Display all the filters in this filter bank and save image as LM.png,
	use command "cv2.imwrite(...)"
	"""
	# Set the number of scales and orientations
	scales = [1,np.sqrt(2),2, 2*np.sqrt(2)]
	orientations = np.arange(180,0,-30)
	filter_size=60


	# Create the filter bank
	lms_filter_bank = create_lm_bank(scales, orientations,filter_size)
	create_filter_img(lms_filter_bank ,'LMS.png',12 , (16,6))

	# Set the number of scales and orientations
	scales = [np.sqrt(2),2,2*np.sqrt(2),4]
	orientations = np.arange(180,0,-30)

	# Create the filter bank
	lml_filter_bank = create_lm_bank(scales, orientations,filter_size)
	create_filter_img(lml_filter_bank ,'LML.png',12 , (16,6))


	"""
	Generate Gabor Filter Bank: (Gabor)
	Display all the filters in this filter bank and save image as Gabor.png,
	use command "cv2.imwrite(...)"
	"""

	# Set the number of scales and orientations
	scales = [1,np.sqrt(2),2,2*np.sqrt(2)]
	scales = [6,10,14,18,20]
	frequencies = [8,6,4,3,2.5]
	orientations = np.arange(0,1,(1/8))*np.pi

	# Create the filter bank
	gabor_filter_bank = create_gabor_bank(scales, orientations, frequencies)
	create_filter_img(gabor_filter_bank ,'Gabor.png',8 , (16,6))


	"""
	Generate Half-disk masks
	Display all the Half-disk masks and save image as HDMasks.png,
	use command "cv2.imwrite(...)"
	"""
	radii = [3,5,7,10]
	orientations = np.linspace(0,360,num = 12)
	half_disk_filter_bank = create_HalfDisc(radii, orientations)
	create_filter_img(half_disk_filter_bank ,'HDMasks.png',8 , (16,6))

	

	"""
	Generate Texton Map
	Filter image using oriented gaussian filter bank
	"""

	print("generating texton maps..")
	
	images = loadImages(image_folder_name, files=None)
	file_names = os.listdir(image_folder_name)
	total_filters = dog_filter_bank
	# total_filters =  dog_filter_bank
	filtered_images = []
	texton_maps = []


	for i,image in enumerate(images):
		filtered_image = []
		for filter in total_filters:	
			greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			filtered_image.append(cv2.filter2D(src=greyscale_image, ddepth=-1, kernel=filter))
		filtered_image = np.array(filtered_image)
		# print(filtered_image.shape)
		filtered_images.append(filtered_image)	

	"""
	Generate texture ID's using K-means clustering
	Display texton map and save image as TextonMap_ImageName.png,
	use command "cv2.imwrite('...)"
	"""

	for i , filtered_image in enumerate(filtered_images):
		# print(filtered_image.shape)
		f,x,y = filtered_image.shape
		input_mat = filtered_image.reshape([f, x*y])
		input_mat = input_mat.transpose()
		# print(input_mat.shape)

		kmeans = sklearn.cluster.KMeans(n_clusters = 64, n_init = 4).fit(input_mat)
		labels = kmeans.predict(input_mat)
		# print(labels.shape)

		texton_image = labels.reshape([x,y])
		# print(texton_image.shape)

		texton_maps.append(texton_image)
		if not os.path.exists(op_folder_name):
			os.makedirs(op_folder_name)
		plt.imsave( op_folder_name +"/TextonMap_"+  file_names[i], texton_image)
	print("texton maps generated")


	"""
	Generate Texton Gradient (Tg)
	Perform Chi-square calculation on Texton Map
	Display Tg and save image as Tg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	textron_gradients = []
	for i,textron_map in enumerate(texton_maps):
		T_g = chisquareDistance(textron_map, 64, half_disk_filter_bank)
		T_g = np.array(T_g)
		T_g = np.mean(T_g, axis = 0)
		textron_gradients.append(T_g)
		plt.imsave( op_folder_name +"/Tg_"+  file_names[i], T_g)


	"""
	Generate Brightness Map
	Perform brightness binning 
	"""
	brightness_maps = []
	for i,image in enumerate(images):
		image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		x,y = image_gray.shape
		input_mat = image_gray.reshape([x*y,1])
		kmeans = sklearn.cluster.KMeans(n_clusters = 16, n_init = 4)
		kmeans.fit(input_mat)
		labels = kmeans.predict(input_mat)
		brightness_image = labels.reshape([x,y])
		brightness_maps.append(brightness_image)
		#plt.imshow(brightness_image)
		#plt.show()
		plt.imsave( op_folder_name +"/BrightonMap_"+  file_names[i], brightness_image) 


	"""
	Generate Brightness Gradient (Bg)
	Perform Chi-square calculation on Brightness Map
	Display Bg and save image as Bg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	brightness_gradients =[]
	print("generating brightness gradient..")
	for i,brightness_map in enumerate(brightness_maps):
		B_g = chisquareDistance(brightness_map, 16, half_disk_filter_bank)
		B_g = np.array(B_g)
		B_g = np.mean(B_g, axis = 0)
		#plt.imshow(B_g)
		#plt.show()
		brightness_gradients.append(B_g)
		plt.imsave( op_folder_name +"/Bg_"+  file_names[i], B_g) 


	"""
	Generate Color Map
	Perform color binning or clustering
	"""
	print("generating color maps..")
	color_maps =[]
	for i,image in enumerate(images):
		x,y,c = image.shape
		input_mat = image.reshape([x*y,c])
	
		kmeans = sklearn.cluster.KMeans(n_clusters = 16, n_init = 4)
		kmeans.fit(input_mat)
		labels = kmeans.predict(input_mat)
		color_image = labels.reshape([x,y])
		color_maps.append(color_image)
		#plt.imshow(color_image)
		#plt.show()			
		plt.imsave( op_folder_name +"/ColorMap_"+  file_names[i], color_image)


	"""
	Generate Color Gradient (Cg)
	Perform Chi-square calculation on Color Map
	Display Cg and save image as Cg_ImageName.png,
	use command "cv2.imwrite(...)"
	"""
	color_gradients= []
	print("generating color gradient..")
	for i,color_map in enumerate(color_maps):
		C_g = chisquareDistance(color_map, 16, half_disk_filter_bank)
		C_g = np.array(C_g)
		C_g = np.mean(C_g, axis = 0)
		#plt.imshow(C_g)
		#plt.show()
		color_gradients.append(C_g)
		plt.imsave( op_folder_name +"/Cg_"+  file_names[i], C_g)


	"""
	Read Sobel Baseline
	use command "cv2.imread(...)"
	"""
	baseline_files = jpg2pngList(file_names)
	print(baseline_files)
	sobel_baseline_folder = ".\BSDS500\SobelBaseline"
	sobel_baseline = loadImages(sobel_baseline_folder, baseline_files)


	"""
	Read Canny Baseline
	use command "cv2.imread(...)"
	"""
	canny_baseline_folder = ".\BSDS500\CannyBaseline"
	canny_baseline = loadImages(canny_baseline_folder, baseline_files)


	"""
	Combine responses to get pb-lite output
	Display PbLite and save image as PbLite_ImageName.png
	use command "cv2.imwrite(...)"
	"""
	print("generating pb lite output..")

	for i in range(len(images)):	
		print("generating edges for image ", baseline_files[i])	
		
		Canny_edge = cv2.cvtColor(canny_baseline[i], cv2.COLOR_BGR2GRAY)
		Sobel_edges = cv2.cvtColor(sobel_baseline[i], cv2.COLOR_BGR2GRAY)
		Feature_info = (textron_gradients[i] + brightness_gradients[i] + color_gradients[i])/3
		w1 = 0.5
		w2 = 0.5
		Baseline_info = (w1 * Canny_edge) + (w2 * Sobel_edges)

		pb_edge = np.multiply(Feature_info, Baseline_info)
		

		plt.imshow(pb_edge, cmap = "gray")
		plt.imsave( op_folder_name +"/PbLite_"+  file_names[i], pb_edge)

		pb_edge = cv2.threshold(pb_edge, 60.0, 255, cv2.THRESH_BINARY)[1]
		plt.imsave( op_folder_name +"/PbLite_wt_"+  file_names[i], pb_edge)

    
if __name__ == '__main__':
    main()
 


