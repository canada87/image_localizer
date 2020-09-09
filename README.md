# image_localizer
This project takes images from a microscope, the images represent lasers spots at the 2 ends of a waveguides.
The images represent 2 hot spots, everytime in a different position.

The goal is localize the position of the 2 hot spots.

For doing so multiple DNN with VGG16, residual, inception architectures are tested.
The prediction is done on an horizontal assemble, where the models of the last 10 epochs are averaged together in order to achieve a more accurate and stable prediction.

Image augmentation is used due to small dataset. All the images are used with multiple rotations, increasing the data set 10 times.

2 other methods are used as benchmark: finding the maximum in the image, looking at feature with the convolution and fit with a gaussian.

Any DNN outperform the benchmark with a resolution 10 times better.
The best results found is with an average error of 0.5 pixels on both direction (xy).

The code takes a gray image as input and gives the xy coordinates of the spots as outputs.

# Files
image_location_ML.py is the main, it takes the images as input, it performs the image augmentation and feeds the images to the training.

lib_monica_ML.py has the classes for the image augmentation and the models.

Immagine1.png is an example of the images used
