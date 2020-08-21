#Isabella Patnode ~ COMP233.A ~ Bash Script for MPI

#!/bin/bash

numNodes=(4 8 12 16)

#runs Jacobi program for each of the #procs in the ary
for node in ${numNodes[@]};
do
	mpicc -o rosie.exe rosie.c
	bccd-syncdir . ~/machines-openmpi
	mpirun -machinefile ~/machines-openmpi -np "$node" --bynode /tmp/node000-bccd/rosie.exe
done 