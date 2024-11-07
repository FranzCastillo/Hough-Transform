all: pgm.o	hough.o

hough.o:	hough.cu pgm.o
	nvcc hough.cu pgm.o -o hough.o

pgm.o:	common/pgm.cpp
	g++ -c common/pgm.cpp -o ./pgm.o
