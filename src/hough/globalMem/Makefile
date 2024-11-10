all: hough.o

hough.o:	hough_global.cu ../../pgm.o
	nvcc -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc hough_global.cu ../../pgm.o -o hough.o
