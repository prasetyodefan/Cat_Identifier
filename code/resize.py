import cv2

# Load the image
imgpath = '..\\Cat_Identifier\\asset\\dataset\\siamese\\siamese (20).jpg'
image = cv2.imread(imgpath, 0)

# Get the dimensions of the image
height, width = image.shape[:2]

# Set the desired dimensions for the resized image
resized_height, resized_width = 500, 500

# Use the cv2.resize() function to resize the image
resized_image = cv2.resize(image, (resized_width, resized_height))

# Save the resized image
cv2.imshow("Image", resized_image)
cv2.imwrite(
    '..\\Cat_Identifier\\asset\\resizeimg\\resizeimg1.jpg', resized_image)
