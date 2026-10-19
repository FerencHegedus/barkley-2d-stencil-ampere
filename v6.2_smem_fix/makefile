CMPL_OPT = -O3 -std=c++11 --ptxas-options=-v --gpu-architecture=sm_86 -lineinfo -maxrregcount=64 --fmad=false --prec-div=true --prec-sqrt=true
SOURCE   = kernel.cu
all: kernel.exe
kernel.exe: $(SOURCE)
	nvcc -o	kernel.exe $(SOURCE) $(CMPL_OPT)

clean:
	rm -f kernel.exe
	rm -f u_sol.txt
	
re:
	rm -f kernel.exe
	rm -f u_sol.txt
	make
	./kernel.exe