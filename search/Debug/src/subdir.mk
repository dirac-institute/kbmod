################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/ImageStack.cpp \
../src/KBMOSearch.cpp \
../src/LayeredImage.cpp \
../src/PointSpreadFunc.cpp \
../src/RawImage.cpp \
../src/kbmod.cpp 

CU_SRCS += \
../src/kernels.cu 

CU_DEPS += \
./src/kernels.d 

OBJS += \
./src/ImageStack.o \
./src/KBMOSearch.o \
./src/LayeredImage.o \
./src/PointSpreadFunc.o \
./src/RawImage.o \
./src/kbmod.o \
./src/kernels.o 

CPP_DEPS += \
./src/ImageStack.d \
./src/KBMOSearch.d \
./src/LayeredImage.d \
./src/PointSpreadFunc.d \
./src/RawImage.d \
./src/kbmod.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/kbmod-usr/cuda-workspace/kbmod/code/kbmodGPU/include" -I/usr/local/cuda-8.0/samples/common/inc -G -g -O0 -Xcompiler -fopenmp -Xcompiler -Wall -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/kbmod-usr/cuda-workspace/kbmod/code/kbmodGPU/include" -I/usr/local/cuda-8.0/samples/common/inc -G -g -O0 -Xcompiler -fopenmp -Xcompiler -Wall -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-8.0/bin/nvcc -I"/home/kbmod-usr/cuda-workspace/kbmod/code/kbmodGPU/include" -I/usr/local/cuda-8.0/samples/common/inc -G -g -O0 -Xcompiler -fopenmp -Xcompiler -Wall -std=c++11 -gencode arch=compute_60,code=sm_60  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-8.0/bin/nvcc -I"/home/kbmod-usr/cuda-workspace/kbmod/code/kbmodGPU/include" -I/usr/local/cuda-8.0/samples/common/inc -G -g -O0 -Xcompiler -fopenmp -Xcompiler -Wall -std=c++11 --compile --relocatable-device-code=true -gencode arch=compute_60,code=compute_60 -gencode arch=compute_60,code=sm_60  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


