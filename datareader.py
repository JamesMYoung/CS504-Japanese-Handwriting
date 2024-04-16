#ETL-4

import bitstring
import numpy as np
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import glob

t56s = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'

hira_to_num ={
	" A": 0,  " I": 1,  " U": 2,  " E": 3,  " O": 4,
	"KA": 5,  "KI": 6,  "KU": 7,  "KE": 8,  "KO": 9,
	"SA": 10, "SI": 11, "SU": 12, "SE": 13, "SO": 14,
	"TA": 15, "TI": 16, "TU": 17, "TE": 18, "TO": 19,
	"NA": 20, "NI": 21, "NU": 22, "NE": 23, "NO": 24,
	"HA": 25, "HI": 26, "HU": 27, "HE": 28, "HO": 29,
	"MA": 30, "MI": 31, "MU": 32, "ME": 33, "MO": 34,
	"YA": 35, "YI": 36, "YU": 37, "YE": 38, "YO": 39,
	"RA": 40, "RI": 41, "RU": 42, "RE": 43,	"RO": 44,
	"WA": 45, "WI": 46, "WU": 47, "WE": 48, "WO": 49,
	" N": 50
}

def read_record_ETL4(f, pos=0):
   f = bitstring.ConstBitStream(filename=f)
   f.bytepos = pos * 2952
   r = f.readlist('2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,15*uint:36,pad:1008,bytes:21888')
   return r
   
#unpacks hiragana from compressed
#file and turns it into an image
def create_hiragana():
	filename = 'ETL4/ETL4C'
	
	X = np.zeros((6112, 76, 72))
	Y = np.zeros((6112))
	
	print("Begin reading hiragana...")
	
	#6112 data entries
	for i in range(6112):
		r = read_record_ETL4(filename, pos=i)
		char_type = ''.join([t56s[c] for c in r[4:8]])
		iF = Image.frombytes('F', (r[18], r[19]), r[-1], 'bit', 4)
		
		Y[i] = hira_to_num[char_type[2:]]
		
		#this is very inefficient, unsure how to do this better
		for x in range(r[18]):
			for y in range(r[19]):
				X[i][y][x] = int(iF.getpixel((x,y)))
		iF = iF.convert('RGB')
		iF.save("basedata/_{}_{}.png".format(char_type[2:],i), format="PNG")
	print("Finished reading hiragana.")
	return X, Y
	
#code taken from /based on the keras blog article
#https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
def increase_images():
	datagen = ImageDataGenerator(
        rotation_range=1,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest')

	for filename in glob.glob('basedata/*.png'):
		img = load_img(filename)  
		img = img.convert('L')
		x = img_to_array(img)
		x = x.reshape((1,) + x.shape)  

		i = 0
		for batch in datagen.flow(x, batch_size=1,
								save_to_dir='hiragana', save_prefix=filename[9:13], save_format='png'):
			i += 1
			if i > 10:
				break  # otherwise the generator would loop indefinitely
	
	
#loads hiragana images into memory to be used
#in the Convolutional Neural Network
def load_hiragana():
	
	X = np.zeros((63038, 76, 72, 1))
	Y = np.zeros((63038))
	
	i = 0
	for filename in glob.glob('hiragana/*.png'):
		img = Image.open(filename)
		img = img.convert('L')
		X[i] = img_to_array(img)
		
		
		Y[i] = hira_to_num[filename[10:12]]
		i = i + 1
		
	return X, Y




	
	