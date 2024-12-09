# RTConvolver
A realtime convolution VST3.
## How it works

![SignalFlow](https://github.com/user-attachments/assets/4eb5a563-39f0-47b4-afc2-9028b1854ef8)

All heavylifting is done on GPU. \
Main inputs(1 & 2) are convolved with side chain inputs (3 & 4)  and will determine 2 outputs.

## How to build
This project is based on cmake.
You will need a working cuda compiler (NVCC).
 ```shell
  git clone https://github.com/zeloe/RTConvolver.git
  cmake . -B build -G "Visual Studio 17 2022"
```

## Known Issues
Sometimes there is some latency. \
Doesn't properly work inside juce audio plugin host. \
Currently supported buffersizes are 512 and 1024. \
Offline rendering doesn't work.
## What works
See it in action [Video](https://youtu.be/UnypuxxIxOg) \
Works in Reaper and Ableton 12.
## Hardware
NVIDIA GTX 1660 TI

## To Do
Add proper kernels using maximum number of threads for each buffersize.

## Download (early build windows)
[Download](https://github.com/zeloe/RTConvolver/releases/tag/v.0.0.2)
