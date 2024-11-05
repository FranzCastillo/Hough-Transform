# Hough Transform
## How to run?
1. Clone the repository
2. Run the following command in the terminal to start the container:
```bash
docker-compose run -it cuda-app /bin/bash
```

## How to compile?
1. For this excercise:
```bash
nvcc -o ht_v1.o hough_v1.cu
./ht_v1.o
```