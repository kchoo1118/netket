CC = g++
ANSIC = gcc -ansi
C99 = gcc -std=c99
CFLAGS = -O3 -I ../
#LFLAGS = ../../fortran/libpfapack.a -llapack -lblas -lgfortran
LFLAGS = ../libpfapack.a -llapack -lblas -lgfortran

all:  example1_c++.out

example1_c++.out: example1_c++.cc
	$(CC) $(CFLAGS) example1_c++.cc -o example1_c++.out ../libcpfapack.a $(LFLAGS)

clean:
	rm -f *.out
