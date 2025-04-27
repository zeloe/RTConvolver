# RTConvolver
RTConvolver is a real-time GPU-powered convolution plugin (VST3) designed for high-performance audio processing.


## Setup in DAW
To set up RTConvolver in your DAW, follow these steps:
1. Create a track with 4 channels and load plugin.
2. Route a stereo signal to channels 1 and 2.
3. Route a different stereo signal to channels 3 and 4.

This allows RTConvolver to process two separate stereo signals simultaneously.

## How to Build Windows
- CMake installed
- Working Cuda Compiler (NVCC)
### Steps to build:
1. Clone the repository:
   ```bash
   git clone https://github.com/zeloe/RTConvolver.git
   ```
2. Run cmake on Windows
```shell
cmake -B build -G "Visual Studio 17 2022"
```
## How to Build Apple Silicon:
- CMake installed
- Xcode with command-line tools installed
### Steps to build:
1. Clone the repository:
   ```bash
   git clone https://github.com/zeloe/RTConvolver.git
   ```
2. Download `metal-cpp`

Download [metal-cpp](https://developer.apple.com/metal/cpp/) from Apple.  
Extract it, and copy the `metal-cpp` folder inside the `metal-cmake` directory:

```text
vst_sources/METAL_sources/metal-cmake/  ← copy metal-cpp here
```
3. Run cmake on Apple Silicon:
```bash
cmake -B build -G Xcode
 ```
## To Do
Find a way to properly normalize output.

## Tested on 
Macbook Air M4 \
GeForce GTX 1660 Ti 

