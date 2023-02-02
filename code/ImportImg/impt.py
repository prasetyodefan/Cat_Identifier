import os

os.chdir('..\\Cat_Identifier\\asset\\dataset\\bengal55')
image_list = os.listdir()
image_list = [a for a in image_list if a.endswith('jpg')]

print(image_list)