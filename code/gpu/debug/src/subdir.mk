################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CU_SRCS += \
../src/main.cu 

CPP_SRCS += \
../src/FakeAsteroid.cpp \
../src/GeneratorPSF.cpp 

OBJS += \
./src/FakeAsteroid.o \
./src/GeneratorPSF.o \
./src/main.o 

CU_DEPS += \
./src/main.d 

CPP_DEPS += \
./src/FakeAsteroid.d \
./src/GeneratorPSF.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_61,code=sm_61  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -G -g -O0 -Xcompiler -fopenmp -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	nvcc -G -g -O0 -Xcompiler -fopenmp -std=c++11 -gencode arch=compute_61,code=sm_61 -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	nvcc -G -g -O0 -Xcompiler -fopenmp -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_61,code=compute_61 -gencode arch=compute_61,code=sm_61  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


