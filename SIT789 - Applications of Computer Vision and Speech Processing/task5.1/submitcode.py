# %% [markdown]
# # Image Recognition Demo

# %% [markdown]
# ## Step 1: Initialise GPU

# %%
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets 
 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# If GPU is available, the print will show 'cuda'. Otherwise it will show 'cpu'
print(device)

# %% [markdown]
# ## Step 2: Prepare dataset

# %%
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', 
                                        train=True, #for training
                                        download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, 
                                          batch_size=4, #process only 4 images at a time
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', 
                                       train=False, #not for training
                                       download=True, transform=transform)

testloader = torch.utils.data.DataLoader(testset, 
                                         batch_size=4, #process only 4 images at a time
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# %% [markdown]
# ## Step 3: Visualise some training images

# %%
# function to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))

# print true labels of the images
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# %% [markdown]
# ## Step 4: Define the network architecture

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# %% [markdown]
# ## Step 5: Set the training parameters and optimizer

# %%
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.AdamW(net.parameters(), lr=0.0005) #lr: learning rate

# %% [markdown]
# ## Step 6: Train the network

# %%
#training

net.to(device)

start_time = time.time()

loss_history = []
epoch = 50
for e in range(epoch):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # set the parameter gradients to zero
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('epoch: %d, baches: %5d, loss: %.3f' %
                  (e + 1, i + 1, running_loss / 2000))
            loss_history.append(running_loss)
            running_loss = 0.0

print('Finished Training')

print('Training time in %s seconds ---' % (time.time() - start_time))

plt.plot(loss_history, label = 'training loss', color = 'r')
plt.legend(loc = 'upper left')
plt.show()

# Save the network after training to file
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

# %% [markdown]
# ## Step 7: Test the network with some test samples

# %%
dataiter = iter(testloader)
images, labels = next(dataiter)

# show images and print ground-truth labels
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

net = Net() #create a network
PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH)) #load the trained network
net.to(device)

outputs = net(images.to(device)) #test the network with inputs as images

_, predicted_labels = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % classes[predicted_labels[j]]
                              for j in range(4)))

# %% [markdown]
# ## Step 8: Evaluate the network on the entire test set

# %%
#load the trained network
net = Net()
net.load_state_dict(torch.load(PATH))

#testing
net.to(device)

start_time = time.time()

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, groundtruth_labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted_labels = torch.max(outputs.data, 1)
        total += groundtruth_labels.size(0)
        correct += (predicted_labels == groundtruth_labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

print("Testing time is in %s seconds ---" % (time.time() - start_time))

# %%
class_correct = list(0. for i in range(10)) 
class_total = list(0. for i in range(10)) 
with torch.no_grad(): 
    for data in testloader: 
        images, groundtruth_labels = data[0].to(device), data[1].to(device) 
        outputs = net(images) 
        _, predicted_labels = torch.max(outputs, 1) 
        c = (predicted_labels == groundtruth_labels).squeeze() 
        for i in range(4): 
            label = groundtruth_labels[i] 
            class_correct[label] += c[i].item() 
            class_total[label] += 1 
 
for i in range(10): 
    print('Accuracy of %5s : %2d %%' % ( 
        classes[i], 100 * class_correct[i] / class_total[i])) 

# %%
# Step 9: Calculate and display the confusion matrix for the test set

all_groundtruth = []
all_predicted = []

with torch.no_grad():
    for data in testloader:
        images, groundtruth_labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted_labels = torch.max(outputs, 1)
        all_groundtruth.extend(groundtruth_labels.cpu().numpy())
        all_predicted.extend(predicted_labels.cpu().numpy())

cm = confusion_matrix(all_groundtruth, all_predicted)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
fig, ax = plt.subplots(figsize=(8, 8))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix for CIFAR-10 Test Set')
plt.show()

# %% [markdown]
# ## Step 10: Train and evaluate a neural network on the FoodImages dataset
# We will use the same architecture as for CIFAR-10, but you are free to experiment with different architectures and learning parameters to maximize performance. The following code will train the model, report the overall accuracy, and display the confusion matrix for the test set.

# %%
# Define trainset, testset, and classes for FoodImages dataset

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset_food = datasets.ImageFolder('FoodImages/Train', transform=transform)
testset_food = datasets.ImageFolder('FoodImages/Test', transform=transform)

trainloader_food = torch.utils.data.DataLoader(trainset_food, batch_size=4, shuffle=True, num_workers=2)
testloader_food = torch.utils.data.DataLoader(testset_food, batch_size=4, shuffle=False, num_workers=2)

food_classes = ('Cakes', 'Pasta', 'Pizza')

# Define the network (same as CIFAR-10, but output size = 3)
class FoodNet(nn.Module):
    def __init__(self):
        super(FoodNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

food_net = FoodNet().to(device)

criterion_food = nn.CrossEntropyLoss().to(device)
optimizer_food = optim.AdamW(food_net.parameters(), lr=0.001)

# Train the network
num_epochs = 30
loss_history_food = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader_food, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer_food.zero_grad()
        outputs = food_net(inputs)
        loss = criterion_food(outputs, labels)
        loss.backward()
        optimizer_food.step()
        running_loss += loss.item()
    loss_history_food.append(running_loss)
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss:.3f}')

# # Save the network after training to file
# PATH = './foodimage_net.pth'
# torch.save(food_net.state_dict(), PATH)

# %%
print('Finished Training FoodNet')
plt.plot(loss_history_food, label='training loss', color='g')
plt.legend(loc='upper left')
plt.show()

# %%
# Evaluate on test set
#load the trained network
food_net = FoodNet()
PATH = './foodimage_net.pth'
food_net.load_state_dict(torch.load(PATH))

#testing
food_net.to(device)

food_net.eval()
correct = 0
total = 0
all_groundtruth_food = []
all_predicted_food = []
with torch.no_grad():
    for data in testloader_food:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = food_net(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_groundtruth_food.extend(labels.cpu().numpy())
        all_predicted_food.extend(predicted.cpu().numpy())

print(f'Accuracy of FoodNet on the test images: {100 * correct / total:.2f} %')

# %%
class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))
with torch.no_grad():
    for data in testloader_food:
        images, groundtruth_labels = data[0].to(device), data[1].to(device)
        outputs = food_net(images)
        _, predicted_labels = torch.max(outputs, 1)
        c = (predicted_labels == groundtruth_labels).squeeze()
        for i in range(len(groundtruth_labels)):
            label = groundtruth_labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(3):
    print('Accuracy of %7s : %2d %%' % (
        food_classes[i], 100 * class_correct[i] / class_total[i]))

# %%
cm_food = confusion_matrix(all_groundtruth_food, all_predicted_food)
disp_food = ConfusionMatrixDisplay(confusion_matrix=cm_food, display_labels=food_classes)
fig, ax = plt.subplots(figsize=(6, 6))
disp_food.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix for FoodImages Test Set')
plt.show()

# %%
# Fine-tuning CIFAR10 model on FoodImages dataset (replace last layer)
PATH = './cifar_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
net.fc3 = nn.Linear(84, 3)
net.to(device)
print(net)

# Set up optimizer and loss for fine-tuning
optimizer_ft = optim.AdamW(net.parameters(), lr=0.0005)
criterion_ft = nn.CrossEntropyLoss().to(device)

# Train the fine-tuned model
num_epochs_ft = 30
loss_history_ft = []
for epoch in range(num_epochs_ft):
    running_loss = 0.0
    for i, data in enumerate(trainloader_food, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer_ft.zero_grad()
        outputs = net(inputs)
        loss = criterion_ft(outputs, labels)
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item()
    loss_history_ft.append(running_loss)
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs_ft}, Loss: {running_loss:.3f}')

print('Finished Fine-tuning')

plt.plot(loss_history_ft, label='fine-tuning loss', color='b')
plt.legend(loc='upper left')
plt.show()

# Evaluate fine-tuned model on FoodImages test set
net.eval()
correct_ft = 0
total_ft = 0
all_groundtruth_ft = []
all_predicted_ft = []
with torch.no_grad():
    for data in testloader_food:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total_ft += labels.size(0)
        correct_ft += (predicted == labels).sum().item()
        all_groundtruth_ft.extend(labels.cpu().numpy())
        all_predicted_ft.extend(predicted.cpu().numpy())

print(f'Accuracy of fine-tuned model on FoodImages test images: {100 * correct_ft / total_ft:.2f} %')

# Confusion matrix for fine-tuned model
cm_ft = confusion_matrix(all_groundtruth_ft, all_predicted_ft)
disp_ft = ConfusionMatrixDisplay(confusion_matrix=cm_ft, display_labels=food_classes)
fig, ax = plt.subplots(figsize=(6, 6))
disp_ft.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix for Fine-tuned FoodImages Test Set')
plt.show()

# Per-class accuracy for fine-tuned model
class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))
with torch.no_grad():
    for data in testloader_food:
        images, groundtruth_labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted_labels = torch.max(outputs, 1)
        c = (predicted_labels == groundtruth_labels).squeeze()
        for i in range(len(groundtruth_labels)):
            label = groundtruth_labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(3):
    print('Accuracy of %7s : %2d %%' % (
        food_classes[i], 100 * class_correct[i] / class_total[i]))

# %% [markdown]
# ## Comparison: Training from Scratch vs Fine-Tuning on FoodImages Dataset
# 
# **Training from Scratch (FoodNet):**
# - The model starts with randomly initialized weights and learns features directly from the FoodImages dataset.
# - Training typically takes longer to converge and may require more epochs to achieve good accuracy.
# - Performance can be limited if the dataset is small or lacks diversity.
# - Per-class accuracy and confusion matrix may show more misclassifications, especially for underrepresented classes.
# 
# **Fine-Tuning (CIFAR10 Pre-trained Net):**
# - The model is initialized with weights learned from the large CIFAR10 dataset, then adapted to FoodImages by replacing the last layer.
# - Training converges faster and often achieves higher accuracy, especially when the target dataset is small.
# - Per-class accuracy and confusion matrix usually show improved results and fewer misclassifications.
# - Transfer learning leverages useful features learned from the source domain, improving generalization.
# 
# **Conclusion:**
# Fine-tuning a pre-trained model on the FoodImages dataset provides better overall and per-class accuracy, faster convergence, and improved generalization compared to training from scratch. This demonstrates the effectiveness of transfer learning, especially when the target dataset is limited in size or diversity.

# %%
# Fine-tuning CIFAR10 model on FoodImages dataset, freezing conv1 and conv2
PATH = './cifar_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
net.fc3 = nn.Linear(84, 3)
net.to(device)
print(net)

# Freeze conv1 and conv2 layers
for param in net.conv1.parameters():
    param.requires_grad = False
for param in net.conv2.parameters():
    param.requires_grad = False

# Update optimizer to only train unfrozen parameters
optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()), lr=0.0005)
criterion_ft = nn.CrossEntropyLoss().to(device)

# Print conv1 and conv2 parameters before training
print('conv1 requires_grad:', net.conv1.weight.requires_grad)
print('conv2 requires_grad:', net.conv2.weight.requires_grad)

# Train the fine-tuned model
num_epochs_ft = 30
loss_history_ft = []
for epoch in range(num_epochs_ft):
    running_loss = 0.0
    for i, data in enumerate(trainloader_food, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer_ft.zero_grad()
        outputs = net(inputs)
        loss = criterion_ft(outputs, labels)
        loss.backward()
        optimizer_ft.step()
        running_loss += loss.item()
    loss_history_ft.append(running_loss)
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs_ft}, Loss: {running_loss:.3f}')

print('Finished Fine-tuning (conv1 & conv2 frozen)')

plt.plot(loss_history_ft, label='fine-tuning loss (fc layers only)', color='m')
plt.legend(loc='upper left')
plt.show()

# Evaluate fine-tuned model on FoodImages test set
net.eval()
correct_ft = 0
total_ft = 0
all_groundtruth_ft = []
all_predicted_ft = []
with torch.no_grad():
    for data in testloader_food:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        total_ft += labels.size(0)
        correct_ft += (predicted == labels).sum().item()
        all_groundtruth_ft.extend(labels.cpu().numpy())
        all_predicted_ft.extend(predicted.cpu().numpy())

print(f'Accuracy of fine-tuned model (fc layers only) on FoodImages test images: {100 * correct_ft / total_ft:.2f} %')

# Confusion matrix for fine-tuned model
cm_ft = confusion_matrix(all_groundtruth_ft, all_predicted_ft)
disp_ft = ConfusionMatrixDisplay(confusion_matrix=cm_ft, display_labels=food_classes)
fig, ax = plt.subplots(figsize=(6, 6))
disp_ft.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix for Fine-tuned FoodImages Test Set (fc layers only)')
plt.show()

# Print conv1 and conv2 parameters after training
print('conv1 requires_grad:', net.conv1.weight.requires_grad)
print('conv2 requires_grad:', net.conv2.weight.requires_grad)

# Per-class accuracy for fine-tuned model (fc layers only)
class_correct = list(0. for i in range(3))
class_total = list(0. for i in range(3))
with torch.no_grad():
    for data in testloader_food:
        images, groundtruth_labels = data[0].to(device), data[1].to(device)
        outputs = net(images)
        _, predicted_labels = torch.max(outputs, 1)
        c = (predicted_labels == groundtruth_labels).squeeze()
        for i in range(len(groundtruth_labels)):
            label = groundtruth_labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1
for i in range(3):
    print('Accuracy of %7s : %2d %%' % (
        food_classes[i], 100 * class_correct[i] / class_total[i]))

# %% [markdown]
# # Object detection demo
# 
# The code for this demo is from https://medium.com/@siromermer/pipeline-for-training-custom-faster-rcnn-object-detection-models-with-pytorch-d506d2423343, https://docs.pytorch.org/tutorials/intermediate/torchvision_tutorial.html, https://docs.pytorch.org/vision/main/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn.html
# 

# %% [markdown]
# ## Step 1: Initialise GPU if available

# %%
import matplotlib.pyplot as plt
import cv2
import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from pycocotools.coco import COCO
import time

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

'''
If there is a GPU, it will display 'cuda'. Otherwise, it will show 'cpu'
'''

# %% [markdown]
# ## Step 2: Define dataset class
# 
# Here we suppose the data is stored in COCO format.

# %%
class AquariumDataset(Dataset): # in COCO format
    def __init__(self, image_dir, annotation_path, transforms=None):
        self.image_dir = image_dir
        self.coco = COCO(annotation_path)
        self.image_ids = list(self.coco.imgs.keys())
        self.transforms = transforms

    # Returns total number of images
    def __len__(self):
        return len(self.image_ids)

    # Fetches a single image and its annotations
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        image = Image.open(image_path).convert('RGB')

        # Load all annotations for this image
        annotation_ids = self.coco.getAnnIds(imgIds=image_id)
        annotations = self.coco.loadAnns(annotation_ids)

        # Extract bounding boxes and labels from annotations
        boxes = []
        labels = []
        for obj in annotations:
            xmin, ymin, width, height = obj['bbox']
            xmax = xmin + width
            ymax = ymin + height
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(obj['category_id'])

        # Convert annotations to PyTorch tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor([obj['area'] for obj in annotations], dtype=torch.float32)
        iscrowd = torch.as_tensor([obj.get('iscrowd', 0) for obj in annotations], dtype=torch.int64)

        boxes = boxes.detach().clone().reshape(-1, 4)

        # Package everything into a target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': image_id,
            'area': area,
            'iscrowd': iscrowd
        }

        # Apply transforms if any were passed
        if self.transforms:
            image = self.transforms(image)

        return image, target

# %% [markdown]
# ## Step 3: Load training and validation data

# %%
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

# Returns a simple transform that converts a PIL image to a PyTorch tensor
def get_transform():
    return ToTensor()

# Load training dataset with transform applied
train_dataset = AquariumDataset(
    image_dir='aquarium/train',
    annotation_path='aquarium/train/_annotations.coco.json',
    transforms=get_transform()
)

# Load validation dataset with same transform
val_dataset = AquariumDataset(
    image_dir='aquarium/valid',
    annotation_path='aquarium/valid/_annotations.coco.json',
    transforms=get_transform()
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# %% [markdown]
# ## Step 4: Get the object category names (in the variable 'class_names'). Each category may be under a supercategory.

# %%
cats = train_dataset.coco.loadCats(train_dataset.coco.getCatIds())
cat_names = [cat['name'] for cat in cats]
print('category names', cat_names)

# Get the number of classes in the dataset (including background)
num_classes = len(train_dataset.coco.getCatIds()) + 1  # +1 for background class

print('number of classes', num_classes)

cat_details = train_dataset.coco.loadCats(train_dataset.coco.getCatIds())[:len(cat_names)]
print('category details', cat_details)

# %% [markdown]
# ## Step 5: Load a pre-trained model (e.g., 'FasterRCNN_ResNet50_FPN_Weights', i.e., Faster RCNN with ResNet50 backbone) for the Faster RCNN

# %%
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

# Get the number of input features for the classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move the model to the specified device (GPU or CPU)
model.to(device)

# %% [markdown]
# ## Step 6: Set learning parameters

# %%
# Get parameters that require gradients (the model's trainable parameters)
params = [p for p in model.parameters() if p.requires_grad]

# Define the optimizer (Stochastic Gradient Descent) with learning rate, momentum, and weight decay
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Set the number of epochs for training
num_epochs = 5

# %% [markdown]
# ## Step 7: Train the Faster RCNN
# 
# Make sure that you have the following files in your working directory
# - utils.py
# - transforms.py
# - coco_eval.py
# - engine.py
# - coco_utils.py
# 
# These files can be found at https://github.com/pytorch/vision/tree/main/references/detection

# %%
from engine import train_one_epoch, evaluate

start_time = time.time()
# Loop through each epoch
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')

    # Train the model for one epoch, printing status every 25 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)  # Using train_loader for training

    # Evaluate the model on the validation dataset
    evaluate(model, val_loader, device=device)  # Using val_loader for evaluation

    # Optionally, save the model checkpoint after each epoch
    torch.save(model.state_dict(), f'model_epoch_{epoch + 1}.pth')

print('Training time in %s seconds ---' % (time.time() - start_time))

# %% [markdown]
# ## Step 8: Test the Faster RCNN with a single image

# %%
from torchvision import models, transforms

# Load the same model architecture with correct number of classes
model = models.detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=num_classes)

model.load_state_dict(torch.load('model_epoch_5.pth'))
model.eval()

# Load image with OpenCV and convert to RGB
img_path = 'aquarium/test/IMG_3164_jpeg_jpg.rf.06637eee0b72df791aa729807ca45c4d.jpg' # CHANGE this to your image path
image_bgr = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

# Transform image
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image_pil).unsqueeze(0)

# Inference
with torch.no_grad():
    predictions = model(image_tensor)

# Parse predictions
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Threshold
threshold = 0.55
for i in range(len(boxes)):
    if scores[i] > threshold:
        box = boxes[i].cpu().numpy().astype(int)
        label = cat_names[labels[i]]
        score = scores[i].item()

        # draw label and score
        text = f'{label}: {score:.2f}'
        cv2.putText(image_bgr, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw rectangle and label
        cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

# Convert BGR to RGB for correct display with matplotlib
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Show image with larger figure size
plt.figure(figsize=(16, 12))  # Increase size as needed
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# %% [markdown]
# ## Step 9: Evaluate the Faster RCNN in the test set

# %%
from engine import evaluate
from coco_eval import CocoEvaluator

# Load test dataset with same transform
test_dataset = AquariumDataset(
    image_dir='aquarium/test',
    annotation_path='aquarium/test/_annotations.coco.json',
    transforms=get_transform()
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

num_images = len(test_dataset)
start_time = time.time()
evaluator = evaluate(model.cuda(), test_loader, device=device)
end_time = time.time()
total_time = end_time - start_time
fps = num_images / total_time
print(f'Testing time (seconds): {total_time:.2f}')
print(f'Testing time (minutes): {total_time/60:.2f}')
print(f'Inference speed (frames-per-second): {fps:.2f}')

# %%
from torchvision.models.detection.faster_rcnn import FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_320_fpn( weights=FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT)

# %%
# Load a pre-trained Faster R-CNN model with ResNet50 backbone and FPN

# Get the number of input features for the classifier head
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move the model to the specified device (GPU or CPU)
model.to(device)

# %%
# Get parameters that require gradients (the model's trainable parameters)
params = [p for p in model.parameters() if p.requires_grad]

# Define the optimizer (Stochastic Gradient Descent) with learning rate, momentum, and weight decay
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Set the number of epochs for training
num_epochs = 5

# %%
from engine import train_one_epoch, evaluate

start_time = time.time()
# Loop through each epoch
for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}')

    # Train the model for one epoch, printing status every 25 iterations
    train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=25)  # Using train_loader for training

    # Evaluate the model on the validation dataset
    evaluate(model, val_loader, device=device)  # Using val_loader for evaluation

    # Optionally, save the model checkpoint after each epoch
    torch.save(model.state_dict(), f'model_mobilenet_epoch_{epoch + 1}.pth')

print('Training time in %s seconds ---' % (time.time() - start_time))

# %%
from torchvision import models, transforms

# Load the same model architecture with correct number of classes
model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=None, num_classes=num_classes)

model.load_state_dict(torch.load('model_mobilenet_epoch_5.pth'))
model.eval()

# Load image with OpenCV and convert to RGB
img_path = 'aquarium/test/IMG_8497_MOV-0_jpg.rf.5c59bd1bf7d8fd7a20999d51a79a12c0.jpg' # CHANGE this to your image path
image_bgr = cv2.imread(img_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
image_pil = Image.fromarray(image_rgb)

# Transform image
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image_pil).unsqueeze(0)

# Inference
with torch.no_grad():
    predictions = model(image_tensor)

# Parse predictions
boxes = predictions[0]['boxes']
labels = predictions[0]['labels']
scores = predictions[0]['scores']

# Threshold
threshold = 0.55
for i in range(len(boxes)):
    if scores[i] > threshold:
        box = boxes[i].cpu().numpy().astype(int)
        label = cat_names[labels[i]]
        score = scores[i].item()

        # draw label and score
        text = f'{label}: {score:.2f}'
        cv2.putText(image_bgr, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Draw rectangle and label
        cv2.rectangle(image_bgr, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

# Convert BGR to RGB for correct display with matplotlib
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# Show image with larger figure size
plt.figure(figsize=(16, 12))  # Increase size as needed
plt.imshow(image_rgb)
plt.axis('off')
plt.show()

# %%
from engine import evaluate
from coco_eval import CocoEvaluator

# Load test dataset with same transform
test_dataset = AquariumDataset(
    image_dir='aquarium/test',
    annotation_path='aquarium/test/_annotations.coco.json',
    transforms=get_transform()
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

num_images = len(test_dataset)
start_time = time.time()
evaluator = evaluate(model.cuda(), test_loader, device=device)
end_time = time.time()
total_time = end_time - start_time
fps = num_images / total_time
print(f'Testing time (seconds): {total_time:.2f}')
print(f'Testing time (minutes): {total_time/60:.2f}')
print(f'Inference speed (frames-per-second): {fps:.2f}')

# %%



