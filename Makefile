LAYERS=layers/*.cu

lenet5: layers/* main_cpu.cpp
	nvcc main_cpu.cpp $(LAYERS) -o lenet5

lenet5_gpu: layers/* main_gpu.cu
	nvcc -DCUDA main_gpu.cu $(LAYERS) -o lenet5_gpu

clean:
	-rm -f lenet5 lenet5_gpu
