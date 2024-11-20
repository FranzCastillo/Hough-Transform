# Hough Transform
## How to create the Environment?
1. Clone the repository
2. Run the following command in the terminal to start the container:
```bash
docker-compose run -it cuda-app /bin/bash
```

Then, we have to install the `OpenCV` library in the container. To do so, run the following commands in the terminal:

```bash
apt update

apt install libopencv-dev
```

Then, to get acces to all the file directories containing the code, acces `hough`
```bash
cd hough
```

## How to Run?
Access the directory of the type of program you want to run
```bash
cd <constantMem|globalMem|sharedMem>
```

Then, run the following command to compile the code:
```bash
bash run.sh
```
This bash script automatically compiles and runs the program, so you should be able to see the compiled file (with the `*.o` extension) and an image that's the result of running said program.

## Side Note
If there's an error related to the `\r` character while trying to run the bash script, run the following command:
```bash
sed -i 's/\r//g' run.sh
```
