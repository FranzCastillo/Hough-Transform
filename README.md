# Hough Transform
## How to run?
1. Clone the repository
2. Run the following command in the terminal to start the container:
```bash
docker-compose run -it cuda-app /bin/bash
```

Then run:
```bash
apt update

apt install libopencv-dev
```

To install the library for generating different image formats

## How to compile?
1. For this excercise:
```bash
make

./hough.o data/runway.pgm
```