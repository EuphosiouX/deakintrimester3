# %%
import numpy as np 
import cv2 as cv 
from sklearn.cluster import KMeans 
import pickle 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn import svm 
from sklearn.ensemble import AdaBoostClassifier 

class Dictionary(object): 
    def __init__(self, name, img_filenames, num_words): 
        self.name = name #name of your dictionary 
        self.img_filenames = img_filenames #list of image filenames 
        self.num_words = num_words #the number of words 
         
        self.training_data = [] #training data used to learn clusters 
        self.words = [] #list of words, which are the centroids of clusters 
     
    def learn(self): 
        sift = cv.SIFT_create() 
         
        num_keypoints = [] #used to store the number of keypoints in each image 
         
        #load training images and compute SIFT descriptors 
        for filename in self.img_filenames: 
            img = cv.imread(filename) 
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
            list_des = sift.detectAndCompute(img_gray, None)[1] 
            if list_des is None: 
                num_keypoints.append(0) 
            else: 
                num_keypoints.append(len(list_des)) 
                for des in list_des: 
                    self.training_data.append(des) 
             
        #cluster SIFT descriptors using K-means algorithm 
        kmeans = KMeans(self.num_words) 
        kmeans.fit(self.training_data) 
        self.words = kmeans.cluster_centers_ 
         
        #create word histograms for training images 
        training_word_histograms = [] #list of word histograms of all training images 
        index = 0 
        for i in range(0, len(self.img_filenames)): #for each file, create a histogram 
            histogram = np.zeros(self.num_words, np.float32) 
            #if some keypoints exist 
            if num_keypoints[i] > 0: 
                for j in range(0, num_keypoints[i]): 
                    histogram[kmeans.labels_[j + index]] += 1 
                index += num_keypoints[i] 
                histogram /= num_keypoints[i] 
                training_word_histograms.append(histogram) 
         
        return training_word_histograms
    
    def create_word_histograms(self, img_filenames): 
        sift = cv.SIFT_create() 
        histograms = [] 
         
        for filename in img_filenames: 
            img = cv.imread(filename) 
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
            descriptors = sift.detectAndCompute(img_gray, None)[1] 
         
            histogram = np.zeros(self.num_words, np.float32) #word histogram  
         
            if descriptors is not None: 
                for des in descriptors: 
                    #find the best matching word 
                    min_distance = 1111111 #this can be any large number 
                    matching_word_ID = -1 #initialise ID with an impractical value 
                     
                    for i in range(0, self.num_words): #find the best matching word 
                        distance = np.linalg.norm(des - self.words[i]) 
                        if distance < min_distance: 
                            min_distance = distance 
                            matching_word_ID = i 
                     
                    histogram[matching_word_ID] += 1 
                 
                histogram /= len(descriptors) #make histogram a prob distribution 
         
            histograms.append(histogram) 
         
        return histograms 

# %%
import os 
 
foods = ['Cakes', 'Pasta', 'Pizza'] 
path = 'FoodImages/' 
training_file_names = [] 
training_food_labels = [] 
for i in range(0, len(foods)): 
    sub_path = path + 'Train/' + foods[i] + '/' 
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)] 
    sub_food_labels = [i] * len(sub_file_names) #create a list of N elements, all are i 
    training_file_names += sub_file_names 
    training_food_labels += sub_food_labels 
     
print(training_file_names) 
print(training_food_labels)

# %%
num_words = 50 
dictionary_name = 'food' 
dictionary = Dictionary(dictionary_name, training_file_names, num_words) 

# %%
training_word_histograms = dictionary.learn()

# %%
#save dictionary 
with open('food_dictionary.dic', 'wb') as f: #'wb' is for binary write 
    pickle.dump(dictionary, f)

# %%
with open('food_dictionary.dic', 'rb') as f: #'rb' is for binary read 
    dictionary = pickle.load(f) 

# %%
test_file_names = [] 
test_food_labels = []

#load test images and create word histograms
for i in range(0, len(foods)): 
    sub_path = path + 'Test/' + foods[i] + '/' 
    sub_file_names = [os.path.join(sub_path, f) for f in os.listdir(sub_path)] 
    sub_food_labels = [i] * len(sub_file_names) #create a list of N elements, all are i 
    test_file_names += sub_file_names 
    test_food_labels += sub_food_labels 
    
word_histograms = dictionary.create_word_histograms(test_file_names) 

# %% [markdown]
# # KNN

# %%
num_nearest_neighbours = 5 #number of neighbours 
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours) 
knn.fit(training_word_histograms, training_food_labels) 
predicted_food_labels = knn.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("num_nearest_neighbour:", num_nearest_neighbours)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
num_nearest_neighbours = 10 #number of neighbours 
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours) 
knn.fit(training_word_histograms, training_food_labels) 
predicted_food_labels = knn.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("num_nearest_neighbour:", num_nearest_neighbours)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
num_nearest_neighbours = 15 #number of neighbours 
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours) 
knn.fit(training_word_histograms, training_food_labels) 
predicted_food_labels = knn.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("num_nearest_neighbour:", num_nearest_neighbours)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
num_nearest_neighbours = 20 #number of neighbours 
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours) 
knn.fit(training_word_histograms, training_food_labels) 
predicted_food_labels = knn.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("num_nearest_neighbour:", num_nearest_neighbours)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
num_nearest_neighbours = 25 #number of neighbours 
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours) 
knn.fit(training_word_histograms, training_food_labels) 
predicted_food_labels = knn.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("num_nearest_neighbour:", num_nearest_neighbours)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
num_nearest_neighbours = 30 #number of neighbours 
knn = KNeighborsClassifier(n_neighbors = num_nearest_neighbours) 
knn.fit(training_word_histograms, training_food_labels) 
predicted_food_labels = knn.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("num_nearest_neighbour:", num_nearest_neighbours)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %% [markdown]
# # SVM

# %%
C = 10
svm_classifier = svm.SVC(C = C, #see slide 32 in week 4 handouts 
kernel = 'linear') #see slide 35 in week 4 handouts 
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = svm_classifier.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("C:", C)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
C = 20
svm_classifier = svm.SVC(C = C, #see slide 32 in week 4 handouts 
kernel = 'linear') #see slide 35 in week 4 handouts 
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = svm_classifier.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("C:", C)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
C = 30
svm_classifier = svm.SVC(C = C, #see slide 32 in week 4 handouts 
kernel = 'linear') #see slide 35 in week 4 handouts 
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = svm_classifier.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("C:", C)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
C = 40
svm_classifier = svm.SVC(C = C, #see slide 32 in week 4 handouts 
kernel = 'linear') #see slide 35 in week 4 handouts 
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = svm_classifier.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("C:", C)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
C = 50
svm_classifier = svm.SVC(C = C, #see slide 32 in week 4 handouts 
kernel = 'linear') #see slide 35 in week 4 handouts 
svm_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = svm_classifier.predict(word_histograms) 
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("C:", C)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %% [markdown]
# # AdaBoost

# %%
n_estimators = 50
adb_classifier = AdaBoostClassifier(n_estimators = n_estimators, #number of weak classifiers 
random_state = 0) 
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = adb_classifier.predict(word_histograms)
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("n_estimators:", n_estimators)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
n_estimators = 100
adb_classifier = AdaBoostClassifier(n_estimators = n_estimators, #number of weak classifiers 
random_state = 0) 
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = adb_classifier.predict(word_histograms)
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("n_estimators:", n_estimators)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
n_estimators = 150
adb_classifier = AdaBoostClassifier(n_estimators = n_estimators, #number of weak classifiers 
random_state = 0) 
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = adb_classifier.predict(word_histograms)
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("n_estimators:", n_estimators)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
n_estimators = 200
adb_classifier = AdaBoostClassifier(n_estimators = n_estimators, #number of weak classifiers 
random_state = 0) 
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = adb_classifier.predict(word_histograms)
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("n_estimators:", n_estimators)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))

# %%
n_estimators = 250
adb_classifier = AdaBoostClassifier(n_estimators = n_estimators, #number of weak classifiers 
random_state = 0) 
adb_classifier.fit(training_word_histograms, training_food_labels)
predicted_food_labels = adb_classifier.predict(word_histograms)
cm = confusion_matrix(test_food_labels, predicted_food_labels) 
print("n_estimators:", n_estimators)
print(cm) 
print(classification_report(test_food_labels, predicted_food_labels))


