import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy, sklearn, urllib
import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import soundfile as sf

app = Flask(__name__)

dic = {0 : 'CHEST', 1 : 'HEAD', 2 : 'MIX'}

model = load_model('mel_model.h5')
model2 = load_model('mfcc_model.h5')

model.make_predict_function()

# 1st prediction
def predict_label(img_path):
	i = image.load_img(img_path, target_size=(256,256))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 256,256,3)
	p = model.predict(i)
	print("[mel] ",p)
	return [p]

# 2nd prediction
def predict_label2(img_path2):
	i = image.load_img(img_path2, target_size=(256,256))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 256,256,3)
	p = model2.predict(i)
	print("[MFCC]   ",p)

	return [p]

@app.route("/<direct_path_file>", methods=['GET'])
def maine(direct_path_file):
	#get the file
	scale_file = 'wavfiles/'+direct_path_file+'.wav'
	# print (scale_file)
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
	plt.savefig("static/" + filename + "_mel.png") 
	# CHEST/f1_arpeggios_belt_c_e.wav_mel.png
	plt.clf()

    # call log mfcc
	x, fs = librosa.load(scale_file)
	mfccs = librosa.feature.mfcc(x, sr=fs)
	mfccs = sklearn.preprocessing.scale(mfccs)
	librosa.display.specshow(mfccs, sr=fs, x_axis='time')
	plt.tight_layout()
	plt.axis('off')
	plt.savefig("static/" + filename + "mfcc.png") 
	# CHEST/f1_arpeggios_belt_c_e.wav_mel.png

	img_path = "static/" + direct_path_file + "_mel.png"
	img_path2 = "static/" + direct_path_file + "mfcc.png"

	#mel PREDICTION
	p = predict_label(img_path)[0]
	#MFCC PREDICTION
	p3 = predict_label2(img_path2)[0]

	print("___________________________________________")

	tentative_max = float("-inf")
	#mel PREDICTION
	pos_idx = -1
	for idx, val in enumerate(p[0]):
		if val > tentative_max:
			tentative_max = val
			pos_idx = idx
				
	tentative_max2 = float("-inf")
	#MFCC PREDICTION
	pos_idx2 = -1
	for idx, val in enumerate(p3[0]):
		if val > tentative_max2:
			tentative_max2 = val
			pos_idx2 = idx

	dic[pos_idx]
	dic[pos_idx2]


	print(p)
	print(tentative_max,dic[pos_idx])
	print(pos_idx)
	print("___________________________________")
	print(p3)
	print(tentative_max2,dic[pos_idx2])
	print(pos_idx2)
	print("\n")
	p1_result = tentative_max*100 
	p3_result = tentative_max2*100
	file1 = direct_path_file + "_mel.png"
	file2 = direct_path_file + "_mfcc.png"

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

	f_result = {}
	answer = str(final_output)
	f_result['output'] = answer		

	return f_result
	# return render_template("index.html", prediction = dic[pos_idx],prediction2 = dic[pos_idx2], img_path = img_path, img_path2 = img_path2,filename = file1, filename2 = file2,percentage_output1 = p1_result,percentage_output2 = p3_result,final_result = final_output)

@app.route("/", methods=['GET', 'POST'])
def maintry():
	
	return render_template("index.html")

if __name__ =='__main__':
	
	app.run(debug = True,host='0.0.0.0')	