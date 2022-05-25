import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import python_speech_features
from IPython import display
import os
import scipy, sklearn, urllib
import timeit
from glob import glob
from numpy import argmax
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image

app = Flask(__name__)

dic = {0 : 'CHEST', 1 : 'HEAD', 2 : 'MIX'}

model = load_model('logmel_model.h5')
model2 = load_model('mfcc_model.h5')

model.make_predict_function()

# #TO COMMENT DIRECT PATH OF WAV FILE
# data_dir = './New folder/'
# scale_file = glob(data_dir + '/*.wav')
# print ("loaded data",scale_file)

# 1st prediction

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(256,256))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 256,256,3)
	p = model.predict(i)
	# p2 = model2.predict(i)

	print("[LOGMEL] ",p)

	return [p]

	# highest_result = dic[argmax(p)]
	# if highest_result < dic[argmax(p2)] :
	# 	highest_result = dic[argmax(p2)]
		
	# return highest_result
		#for image
	# return dic[argmax(p)],dic[argmax(p2)]
	# 2nd Prediction
def predict_label2(img_path2):
	i = image.load_img(img_path2, target_size=(256,256))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 256,256,3)
	p = model2.predict(i)
	# p2 = model2.predict(i)

	print("[MFCC]   ",p)

	return [p]

	# return dic[argmax(p)],dic[argmax(p2)]

	# highest_result2 = dic[argmax(p)]
	# if highest_result2 < dic[argmax(p2)] :
	# 	highest_result2 = dic[argmax(p2)]
		
	# return highest_result2	
	# return highest_acc_result

	# return [p,p2]
	
	
	
	#for image
	# return dic[argmax(p2)]
	
	#array out put
	# return p

#TO COMMENT EXTRACTION OF WAV FILE

# def Extract_to_logmel():
# 	print("scalefile",scale_file[0])
# 	y, sr = librosa.load(scale_file[0])
# 	ps = librosa.feature.melspectrogram(y=y, sr=sr)
# 	ps_db= librosa.power_to_db(ps, ref=1.0)
# 	librosa.display.specshow(ps_db, x_axis='s', y_axis='log')
# 	plt.tight_layout()
# 	plt.axis('off')
# 	logmel_file = (data_dir+'.wav')
# 	plt.savefig("static/" + "test" + "_logmel.png") 
# 	plt.clf()

# def extraction_to_mfcc():
# 	x, fs = librosa.load(scale_file)
# 	mfccs = librosa.feature.mfcc(x, sr=fs)
# 	mfccs = sklearn.preprocessing.scale(mfccs)
# 	librosa.display.specshow(mfccs, sr=fs, x_axis='time')
# 	plt.tight_layout()
# 	plt.axis('off')

# def save_mfcc():
# 	plt.savefig("static/" + mfccs_file + "mfcc.png")
# 	#nakuha mo na yung png 

# extraction_to_mfcc()
# extract_to_logmel()
# # mfccs_file = glob(data_dir+'.wav')
# logmel_file = glob(data_dir+'.wav')

# save_mfcc()
# routes

# @app.route("/", methods=['GET', 'POST'])
# def main():
# 	return render_template("index.html")

# def show_result():
# 	static_dir = "./static/"
# 	img = glob(static_dir + "/.png")
# 	img_path = "static/" + img
# 	img.save(img_path)
# 	p = predict_label(img_path)
# 	return render_template("index.html", prediction = p, img_path = img_path)



# @app.route("/", methods=['GET', 'POST'])
# def extractions():
# 	Extract_to_logmel()
# 	static_dir = "./static/"
# 	# img = glob(static_dir + "/.png")
# 	img_path = "static/" + "test" + "_logmel.png"
# 	# img.save(img_path)
# 	p = predict_label(img_path)
# 	# return render_template("index.html")
# 	# return render_template("index.html", prediction = p, img_path = img_path)
# 	return render_template("index.html", prediction = p, img_path = img_path)





@app.route("/<direct_path_file>", methods=['GET', 'POST'])
def maine(direct_path_file):
	scale_file = 'VocalSet/FULL/female6/arpeggios/belt/'+direct_path_file+'.wav'
	print (scale_file)
	filename = direct_path_file
    # load file
	plt.switch_backend('agg')
    # call log mel spectrogram
	y, sr = librosa.load(scale_file)
	ps = librosa.feature.melspectrogram(y=y, sr=sr)
	ps_db= librosa.power_to_db(ps, ref=1.0)
	librosa.display.specshow(ps_db, x_axis='s', y_axis='log')
	plt.tight_layout()
	plt.axis('off')
	plt.savefig("static/" + filename + "_logmel.png") # CHEST/f1_arpeggios_belt_c_e.wav_logmel.png
	plt.clf()

    # call log mfcc
	x, fs = librosa.load(scale_file)
	mfccs = librosa.feature.mfcc(x, sr=fs)
	mfccs = sklearn.preprocessing.scale(mfccs)
	librosa.display.specshow(mfccs, sr=fs, x_axis='time')
	plt.tight_layout()
	plt.axis('off')
	plt.savefig("static/" + filename + "mfcc.png") 
	# CHEST/f1_arpeggios_belt_c_e.wav_logmel.png

	# img = glob(direct_path_file + "_logmel.png")

	# h_logmel = predict_label(img_path)
	# h_mfcc = predict_label2(img_path2)

	# h_result = h_logmel

	# if h_result < h_mfcc:
	# 	h_result = h_mfcc

	

	img_path = "static/" + direct_path_file + "_logmel.png"
	img_path2 = "static/" + direct_path_file + "mfcc.png"

	# print(img_path)
	# print(img_path2)

	p = predict_label(img_path)[0]
	# p2 = predict_label(img_path)[0]
	p3 = predict_label2(img_path2)[0]
	# p4 = predict_label(img_path2)[0]

	print("___________________________________________")

	tentative_max = float("-inf")

	pos_idx = -1

	# print(p[0],p2[0])

	for idx, val in enumerate(p[0]):
		if val > tentative_max:
			tentative_max = val
			pos_idx = idx
				
	# for idx, val in enumerate(p2[0]):
	# 	if val > tentative_max:
	# 		tentative_max = val
	# 		pos_idx = idx

	dic[pos_idx]

	# print(tentative_max)

	tentative_max2 = float("-inf")

	pos_idx2 = -1

	# print(p3[0],p4[0])

	for idx, val in enumerate(p3[0]):
		if val > tentative_max2:
			tentative_max2 = val
			pos_idx2 = idx
				
	# for idx, val in enumerate(p4[0]):
	# 	if val > tentative_max2:
	# 		tentative_max2 = val
	# 		pos_idx2 = idx

	
	dic[pos_idx]
	dic[pos_idx2]
	print(p)
	print(tentative_max,dic[pos_idx])
	print(pos_idx)

	print("___________________________________")
	print(p3)
	print(tentative_max2,dic[pos_idx2])
	print(pos_idx2)

	# h_result = tentative_max
	# if tentative_max2 > h_result:
	# 	h_result = tentative_max2
	
	print("\n")
	# print(h_result)

	# h_result = p

	# if h_result < p2
	p1_result = tentative_max*100 
	p3_result = tentative_max2*100


	file1 = direct_path_file + "_logmel.png"
	file2 = direct_path_file + "_mfcc.png"

	# chest_result = 0
	# head_result = 0
	# mix_result = 0
	# mix_notice = 0
	# mix_notice2 = 0


	# CHEST AND CHEST RESULT
	if pos_idx == 0 and pos_idx2 == 0:
		chest_result = "CHEST"
		final_output = chest_result
	# #HEAD AND HEAD RESULT
	if pos_idx == 1 and pos_idx2 == 1:
		head_result = "HEAD"
		final_output = head_result
	# #MIX AND MIX RESULT
	if pos_idx == 2 and pos_idx2 == 2:
		mix_result = "MIX"
		final_output = mix_result
	# #CHEST AND HEAD RESULT
	if pos_idx == 0 and pos_idx2 == 1:
		mix_notice = "Note: The application noticed that you changed your voice placement at the recording, meaning you are using two voice placements at a time (head and chest voice). To have an accurate result try to record again."
		final_output = mix_notice
	# #HEAD AND CHEST RESULT
	if pos_idx == 1 and pos_idx2 == 0:
		mix_notice = "Note: The application noticed that you changed your voice placement at the recording, meaning you are using two voice placements at a time (head and chest voice). To have an accurate result try to record again."
		final_output = mix_notice
	# #CHEST AND MIX RESULT
	if pos_idx == 0 and pos_idx2 == 2:
		mix_notice2 = "Note: The application noticed that you used a chest and mixed voice. This happened because you hit the voice placement you expected. However, your voice placement changed. Try to record again to have accurate results."
		final_output = mix_notice2
	# #MIX AND CHEST RESULT
	if pos_idx == 2 and pos_idx2 == 0:
		mix_notice2 = "Note: The application noticed that you used a chest and mixed voice. This happened because you hit the voice placement you expected. However, your voice placement changed. Try to record again to have accurate results."
		final_output = mix_notice2
	# #HEAD AND MIX RESULT
	if pos_idx == 1 and pos_idx2 == 2:
		mix_notice2 = "Note: The application noticed that you used a chest and mixed voice. This happened because you hit the voice placement you expected. However, your voice placement changed. Try to record again to have accurate results."
		final_output = mix_notice2
	# #MIX AND HEAD RESULT
	if pos_idx == 2 and pos_idx2 == 1:
		mix_notice2 = "Note: The application noticed that you used a chest and mixed voice. This happened because you hit the voice placement you expected. However, your voice placement changed. Try to record again to have accurate results."
		final_output = mix_notice2
	


	

	# print("RESULT TRIAL:",chest_result)
	


	# return str(tentative_max) + "___" + str(tentative_max2)
	
		# return str(argmax(p))
		# return str(p)
	return render_template("index.html", prediction = dic[pos_idx],prediction2 = dic[pos_idx2], img_path = img_path, img_path2 = img_path2,filename = file1, filename2 = file2,percentage_output1 = p1_result,percentage_output2 = p3_result,final_result = final_output)

@app.route("/", methods=['GET', 'POST'])
def maintry():
	
	
	return render_template("index.html")

#test_logmel.png
@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

		# return str(argmax(p))
		# return str(p)

	return render_template("index.html", prediction = p, img_path = img_path,imge = img)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)	


#TRIAL LANG TO FOR DIRECT FILE