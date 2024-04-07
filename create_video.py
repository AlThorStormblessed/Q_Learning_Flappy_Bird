# importing libraries 
import os
import cv2 
from PIL import Image 

# Checking the current directory path 
print(os.getcwd()) 

# Folder which contains all the images 
# from which video is to be generated 
path = f"num_episodes_30000/"

mean_height = 0
mean_width = 0

num_of_images = len(os.listdir(path)) 
# print(num_of_images) 

for file in os.listdir(path): 
	im = Image.open(os.path.join(path, file)) 
	width, height = im.size 
	mean_width += width 
	mean_height += height 
	# im.show() # uncomment this for displaying the image 

# Finding the mean height and width of all images. 
# This is required because the video frame needs 
# to be set with same width and height. Otherwise 
# images not equal to that width height will not get 
# embedded into the video 
mean_width = int(mean_width / num_of_images) 
mean_height = int(mean_height / num_of_images) 
 
def generate_video(): 
	images = [img for img in os.listdir(path)] 
	
	# Array images should only consider 
	# the image files ignoring others if any 
	images = sorted(images, key= lambda x: int(x[:-4]))
	print(images)

	frame = cv2.imread(os.path.join(path, images[0])) 

	# setting the frame width, height width 
	# the width, height of first image 
	height, width, layers = frame.shape 

	video = cv2.VideoWriter(
    f"{path[:-1]}.mp4",
    cv2.VideoWriter_fourcc(*"MP4V"),
    24,
    (width, height),
)

	# Appending the images to the video one by one 
	for image in images: 
		video.write(cv2.imread(os.path.join(path, image))) 
	
	# Deallocating memories taken for window creation 
	cv2.destroyAllWindows() 
	video.release() # releasing the video generated 


# Calling the generate_video function 
generate_video() 
