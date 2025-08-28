# %%
import numpy as np 
import matplotlib.pyplot as plt 
import librosa 
import librosa.display 
import IPython.display as ipd 
from pydub import AudioSegment 
from pydub.utils import mediainfo 
speech = AudioSegment.from_wav('arctic_a0005.wav') # Read audio data from file 
x = speech.get_array_of_samples()  # samples x(t) 
x_sr = speech.frame_rate  # sampling rate f - see slide 24 in week 7 lecture slides 
print('Sampling rate: ', x_sr) 
print('Number of samples: ', len(x)) 

# %%
duration = librosa.get_duration(path = 'arctic_a0005.wav') 
n_samples = duration * x_sr 
print('duration: ', duration) 
print('n_samples: ', n_samples)

# %%
x_range = np.linspace(0, duration, len(x)) 
 
plt.figure(figsize = (15, 5)) 
plt.plot(x_range, x) 
plt.xlabel('Time in seconds') 
plt.ylabel('Amplitude') 

# %%
mid_point = int(len(x) / 2) 
x1 = x[0:mid_point] 
x2 = x[mid_point:len(x)] 
 
x1_audio = AudioSegment( 
                data = x1, #raw data 
                sample_width = 2, #2 bytes = 16 bit samples 
                frame_rate = x_sr, #frame rate     
                channels = 1) #channels = 1 for mono and 2 for stereo 
 
x2_audio = AudioSegment( 
                data = x2, #raw data 
                sample_width = 2, #2 bytes = 16 bit samples 
                frame_rate = x_sr, #frame rate 
                channels = 1) #channels = 1 for mono and 2 for stereo 
 
x1_audio.export('arctic_a0005_1.wav', format = 'wav') 
x2_audio.export('arctic_a0005_2.wav', format = 'wav')

# %%
duration1 = librosa.get_duration(path = 'arctic_a0005_1.wav') 
n_samples1 = duration1 * x_sr 
print('duration_1: ', duration1) 
print('n_samples_1: ', n_samples1) 

# %%
x_range1 = np.linspace(0, duration1, len(x1)) 
 
plt.figure(figsize = (15, 5)) 
plt.plot(x_range1, x1) 
plt.xlabel('Time in seconds') 
plt.ylabel('Amplitude') 

# %%
duration2 = librosa.get_duration(path = 'arctic_a0005_2.wav') 
n_samples2 = duration2 * x_sr 
print('duration_2: ', duration2) 
print('n_samples_2: ', n_samples2) 

# %%
x_range2 = np.linspace(0, duration2, len(x2)) 
 
plt.figure(figsize = (15, 5)) 
plt.plot(x_range2, x2) 
plt.xlabel('Time in seconds') 
plt.ylabel('Amplitude') 

# %%
#range of frequencies of interest for speech signal.  
#It can be any positive value, but should be a power of 2 
freq_range = 1024  
#window size: the number of samples per frame. Each frame is of 30ms = 0.03 sec 
win_length = int(x_sr * 0.03) 
#number of samples between two consecutive frames, by default it is set to win_length/4 
hop_length = int(win_length / 2) 
#windowing technique 
window = 'hann' 
X = librosa.stft(np.float32(x),  
n_fft = freq_range,  
window = window,  
hop_length = hop_length,  
win_length = win_length)

# %%
print(X.shape) 

# %%
plt.figure(figsize = (15, 5)) 
#convert the amplitude to decibels, just for illustration purpose 
Xdb = librosa.amplitude_to_db(abs(X)) 
librosa.display.specshow(Xdb, #spectrogram                         
sr = x_sr, #sampling rate 
x_axis = 'time', #label for horizontal axis     
y_axis = 'linear', #presentation scale 
hop_length = hop_length) #hop_length 

# %%
X1 = librosa.stft(np.float32(x1),  
n_fft = freq_range,  
window = window,  
hop_length = hop_length,  
win_length = win_length) 

plt.figure(figsize = (15, 5)) 
#convert the amplitude to decibels, just for illustration purpose 
Xdb1 = librosa.amplitude_to_db(abs(X1)) 
librosa.display.specshow(Xdb1, #spectrogram                         
sr = x_sr, #sampling rate 
x_axis = 'time', #label for horizontal axis     
y_axis = 'linear', #presentation scale 
hop_length = hop_length) #hop_length 

# %%
X2 = librosa.stft(np.float32(x2),  
n_fft = freq_range,  
window = window,  
hop_length = hop_length,  
win_length = win_length) 

plt.figure(figsize = (15, 5)) 
#convert the amplitude to decibels, just for illustration purpose 
Xdb2 = librosa.amplitude_to_db(abs(X2)) 
librosa.display.specshow(Xdb2, #spectrogram                         
sr = x_sr, #sampling rate 
x_axis = 'time', #label for horizontal axis     
y_axis = 'linear', #presentation scale 
hop_length = hop_length) #hop_length 

# %%
#number of samples 
N = 600 
#sample spacing 
T = 1.0 / N 
t = np.linspace(0.0, N*T, N) 
s1 = np.sin(50.0 * 2.0 * np.pi * t) 
s2 = 0.5 * np.sin(80.0 * 2.0 * np.pi * t) 
s = s1 + s2 

# %%
plt.figure(figsize = (15, 5)) 
plt.plot(s1, label = 's1', color = 'r') 
plt.plot(s2, label = 's2', color = 'g') 
plt.plot(s, label = 's', color = 'b') 
plt.xlabel('Samples') 
plt.ylabel('Amplitude') 
plt.legend(loc = "upper left") 

# %%
S = librosa.stft(s, n_fft = N, window = 'hann', hop_length = N, win_length = N) 
S_0 = S[:, 0] 
mag_S_0 = np.abs(S_0) 
plt.plot(mag_S_0, color = 'b') 

# %%
#we define a window length m with fewer samples 
m = 400 
S = librosa.stft(s, n_fft = N, window = 'boxcar', hop_length = int(m/2), win_length = m) 
#we take S_1, which is an intermediate frame. 
S_1 = S[:, 1] 
mag_S_1 = np.abs(S_1) 
plt.plot(mag_S_1, color = 'b') 

# %%
#we define a window length m with fewer samples 
m = 400 
S = librosa.stft(s, n_fft = N, window = 'hann', hop_length = int(m/2), win_length = m) 
#we take S_1, which is an intermediate frame. 
S_1 = S[:, 1] 
mag_S_1 = np.abs(S_1) 
plt.plot(mag_S_1, color = 'b') 

# %%
#we define a window length m with fewer samples 
m = 400 
S = librosa.stft(s, n_fft = N, window = 'hamming', hop_length = int(m/2), win_length = m) 
#we take S_1, which is an intermediate frame. 
S_1 = S[:, 1] 
mag_S_1 = np.abs(S_1) 
plt.plot(mag_S_1, color = 'b') 


