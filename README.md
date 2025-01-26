# RTConvolver
A realtime GPU powered convolution VST3. 
##
### Setup in DAW
In a DAW of your  choice you need to setup a track with 4 channels. \
You should then send on channel 1 & 2 a stereo signal and on channel 3 & 4 a different stereo signal.
### How to build
You will need a working CUDA compiler and cmake. 
 ```shell
  git clone https://github.com/zeloe/RTConvolver.git
  cmake . -B build -G "Visual Studio 17 2022"
```
### Processing
At each call of process block 4 blocks of audio get into plugin. Then each of these audio blocks is on CPU. These audio blocks are then moved on GPU. 
On GPU time a domain delay line is implemented for  each block 1 and 2. By following this logic. 
```mermaid 
graph LR
First;
```
```mermaid 
graph LR
Second-->First;
```
Another different time delay line is implemented on 3 and 4.  By following this logic.
```mermaid 
graph LR
First;
```
```mermaid 
graph LR
First-->Second;
```
Then linear Convolution (without FFT) is done.  Convolving block 1 with block 3 will produce left channel. Convolving block 2 with block 4 will produce right channel. 
These two blocks are then moved back to CPU and you can hear results. 
Using this method you can change e.g. your impulse response in realtime and use other effects on it. 
##
### Convolution Sizes
There is a menu where you can choose how big your convolution size is in seconds. 
- 0.5, 1.0, 2.0, 3.0, 4.0
##
### Volume Knob
This knob is used to scale output volume.
##
### 

## Known Issues
Sometimes there is some latency. \
Doesn't properly work inside juce audio plugin host. \
Currently supported buffersizes are 128 , 256 , 512 , 1024. \
Offline rendering doesn't work.
## What works
How to use it : [Video](https://youtu.be/P2fRFk7yA3U) \
Works in Reaper and Ableton 12.
## Hardware
NVIDIA GTX 1660 TI

## To Do
Add proper kernels using maximum number of threads for each buffersize. \
Find proper way to normalize output.

## Download (early build windows)
[Download](https://github.com/zeloe/RTConvolver/releases/tag/v.0.0.3)
