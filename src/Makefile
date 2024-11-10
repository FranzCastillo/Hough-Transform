all: pgm.o	hough.o

hough.o:	hough.cu pgm.o
	nvcc -I/usr/include/opencv4 -L/usr/lib -lopencv_core -lopencv_imgcodecs -lopencv_imgproc hough.cu pgm.o -o hough.o

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
