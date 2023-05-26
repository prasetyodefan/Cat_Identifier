# Loading input image from file
# Our image initially is in RGB format
# But now we open it in BGR format as function 'cv2.imread' opens it so
x_input = cv2.imread('/content/coba_2.png')

# Getting image shape
print(x_input.shape)  # (1050, 1680, 3)

# Getting blob from input image
# The 'cv2.dnn.blobFromImage' function returns 4-dimensional blob
# from input image after normalizing, and RB channels swapping
# Resulted shape has number of images, number of channels, width and height
# E.G.: blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, swapRB=True)
blob = cv2.dnn.blobFromImage(x_input, 1 / 255.0, (32, 32), swapRB=True, crop=False)

# Getting blob's shape
print(blob.shape)  # (1, 3, 32, 32)


# Preprocessing image in the same way as it was done for training data
# Normalizing by 255.0 we already did in blob, now we need subtract mean image and divide by std image

# Opening file for reading in binary mode
with open('/content/mean_and_std.pickle', 'rb') as f:
    mean_and_std = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3

# Getting mean image and std from dictionary
mean_image = mean_and_std['mean_image']
std = mean_and_std['std']

# Getting shape
print(mean_image.shape)  # (32, 32, 3)
print(std.shape)  # (32, 32, 3)

# Transposing mean and std to make channels come first as we have it now in blob image
mean_image = mean_image.transpose(2, 0, 1)
std = std.transpose(2, 0, 1)

# Getting shape
print(mean_image.shape)  # (3, 32, 32)
print(std.shape)  # (3, 32, 32)
print()

# Subtracting mean image from blob
blob[0] -= mean_image
# Dividing by standard deviation
blob[0] /= std


labels = ['helmet', 'Non_Helmet']

# Opening file for reading in binary mode
with open('/content/model_params_ConvNet1.pickle', 'rb') as f:
    d_trained = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3

# Creating instance of class and initializing model
model = ConvNet1()

# Assigning to the new model loaded parameters from 'pickle' file
model.params = d_trained

# Showing assigned parameters
for i, j in model.params.items():
    print(i + ':', j.shape)
print()
    
# Getting scores from forward pass of input image
# Measuring at the same time execution time
start = timer()
scores = model.scores_for_predicting(blob)
end = timer()

# Scores is given for each image with 10 numbers of predictions for each class
# Getting only one class for each image with maximum value
print('Predicted label is', labels[np.argmax(scores, axis=1)[0]])
print('Time spent for pedicting: {} seconds'.format(round(end - start, 5)))
print()
# Now we have each image with its only one predicted class (index of each row)
# but not with 10 numbers for each class

# Printing all scores
print(scores)

%matplotlib inline
plt.rcParams['figure.figsize'] = (10.0, 8.0) # Setting default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Plotting scores
fig = plt.figure()
x_positions = np.arange(len(labels))
barlist = plt.bar(x_positions, scores[0], align='center', alpha=0.6)
barlist[np.argmax(scores)].set_color('red')
plt.xticks(x_positions, labels, fontsize=15)
plt.ylabel('Value', fontsize=15)
plt.title('Classification of user\'s image', fontsize=20)

plt.show()
# Saving plot
fig.savefig('Classification_of_users_image.png')
plt.close()


# Showing image with predicted label
# Resizing image
x_input = cv2.resize(x_input, (400, 300), interpolation=cv2.INTER_AREA)

# Preparing text with label and score
text = 'Label: {}'.format(labels[np.argmax(scores, axis=1)[0]])

# Preparing colour
colour = [0, 255, 0]

# Putting text with label and confidence on the original image
cv2.putText(x_input, text, (10, 25), cv2.FONT_HERSHEY_TRIPLEX, 0.8, colour, 1)

# Showing resulted image
fig = plt.figure()
plt.imshow(cv2.cvtColor(x_input, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.show()
# Saving plot
fig.savefig('Users_image_with_label.png')
plt.close()