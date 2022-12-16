import cv2 as cv

# Load the image
image = cv.imread('otsus.jpg')

# Get the dimensions of the image
height, width = image.shape[:2]

# Set the desired dimensions for the resized image
resized_height, resized_width = 300, 300

# Use the cv2.resize() function to resize the image
resized_image = cv.resize(image, (resized_width, resized_height))

# Save the resized image
cv.imwrite(
    'G:\\SKRIPSI\\Project_Code\\CatIdentifier\\asset\\resizeimg.jpg', resized_image)
