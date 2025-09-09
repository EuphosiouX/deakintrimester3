# %%
import numpy as np 
import librosa 
from pydub import AudioSegment 
from pydub.utils import mediainfo 
from sklearn import preprocessing 

def mfcc_extraction(audio_filename, #.wav filename 
                    hop_duration, #hop_length in seconds, e.g., 0.015s (i.e., 15ms) 
                    num_mfcc #number of mfcc features 
                    ):  
    speech = AudioSegment.from_wav(audio_filename) #Read audio data from file 
    samples = speech.get_array_of_samples() #samples x(t) 

    sampling_rate = speech.frame_rate #sampling rate f 
    mfcc = librosa.feature.mfcc( 
        y = np.float32(samples),  
        sr = sampling_rate,  
        hop_length = int(sampling_rate * hop_duration),  
        n_mfcc = num_mfcc) 

    return mfcc.T 

# %%
from sklearn.mixture import GaussianMixture 
def learningGMM(features, #list of feature vectors, each feature vector is an array 
                n_components, #the number of components 
                max_iter #maximum number of iterations 
                ): 
    gmm = GaussianMixture(n_components = n_components, max_iter = max_iter) 
    gmm.fit(features) 
    return gmm 

# %%
import os 
path = 'SpeakerData/' 
speakers = os.listdir(path + 'Train/') 
print(speakers)

# %%
from sklearn import preprocessing 

mfcc_all_speakers = [] #list of the MFCC features of the training data of all speakers 
hop_duration = 0.015 #15ms 
num_mfcc = 12 

for s in speakers: 
    sub_path = path + 'Train/' + s + '/' 
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)] 
    mfcc_one_speaker = np.asarray(()) 
    for fn in sub_file_names: 
        mfcc_one_file = mfcc_extraction(fn, hop_duration, num_mfcc) 
        if mfcc_one_speaker.size == 0: 
            mfcc_one_speaker = mfcc_one_file 
        else: 
            mfcc_one_speaker = np.vstack((mfcc_one_speaker, mfcc_one_file)) 
    mfcc_all_speakers.append(mfcc_one_speaker)

# %%
import pickle 
for i in range(0, len(speakers)): 
    with open('TrainingFeatures/' + speakers[i] + '_mfcc.fea','wb') as f: 
        pickle.dump(mfcc_all_speakers[i], f)

# %%
n_components = 5 
max_iter = 50 

gmms = [] #list of GMMs, each is for a speaker 
for i in range(0, len(speakers)): 
    gmm = learningGMM(mfcc_all_speakers[i],  
                    n_components,  
                    max_iter) 
    gmms.append(gmm)

# %%
for i in range(len(speakers)): 
    with open('Models/' + speakers[i] + '.gmm', 'wb') as f: #'wb' is for binary write 
        pickle.dump(gmms[i], f) 

# %%
gmms = [] 
for i in range(len(speakers)): 
    with open('Models/' + speakers[i] + '.gmm', 'rb') as f: #'wb' is for binary write 
        gmm = pickle.load(f) 
        gmms.append(gmm) 

# %%
def speaker_recognition(audio_file_name, gmms):
    # Extract MFCC features from the input audio file
    f = mfcc_extraction(audio_file_name, hop_duration, num_mfcc)

    # Compute the likelihood for each GMM
    scores = [gmm.score(f) for gmm in gmms]

    # Find the index (speaker_id) of the highest score
    speaker_id = int(np.argmax(scores))
    return speaker_id

# %%
speaker_id = speaker_recognition('SpeakerData/Test/Ara/a0522.wav', gmms) 
print(speakers[speaker_id]) 

# %%
from sklearn.metrics import confusion_matrix, accuracy_score

# Prepare test data and ground truth labels
y_true = []
y_pred = []

for i, s in enumerate(speakers): 
    sub_path = path + 'Test/' + s + '/' 
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)] 
    for fn in sub_file_names: 
        pred_id = speaker_recognition(fn, gmms)
        y_true.append(i)
        y_pred.append(pred_id)

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f'Overall recognition accuracy: {accuracy * 100:.2f}%')

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print('Confusion Matrix:')
print(cm)



# %%



