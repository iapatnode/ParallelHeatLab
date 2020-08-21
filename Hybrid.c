/*************************************************************************
*Isabella Patnode ~ COMP233.A ~ Hybrid Jacobi Iteration for Heat Distribution
**********Code originally taken from Argonne National Laboratory**********
*************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <omp.h>

//values of boundary cells
const int NORTH = 100;
const int SOUTH = 100;
const int WEST = 0;
const int EAST = 0;

const int MAX_DIM = 1000; //dimensions of ary
const int MAX_ITER = 500000; //max num of iterations
const int EXTRA_ROWS = 2; //additional rows from neighboring processes

//value approx. must be less than to converge
const float EPSILON = 1.0e-2;
const int NGH_CELL = 4; //num of neighboring cells

const int MAX_RGB = 255; //max RGB value

/* variable used to calc. when to print 
** current iter. and diffNorm */
const int PRINT_ITER = 1000;

//num tuples in one row of ppm file
const int FILE_COLS = 5;

//variable for number of threads
const int NUM_THREADS = 4;

int main(int argc, char* argv[]) {
    int commSize; //number of processes
    int rank; //process ID of current process
    int arySize; //the size of the array
    int remainderRows; //the rows leftover when MAX_DIM is not divisible by commSize
    int r, c, proc; //loop variables
    int iterCount; //tracks number of iterations
    int recvSize; //size of array that is received by process
    float diffNorm, gDiffNorm; //variables for Jac. calc.
    float* temp; //variable for array swapping
    double startTime, stopTime; //variables for timing
    int numFileCols; //tracks num of vals in one row of file
    int red, blue, green, rgb; //vals for interpolation
    float* currentAry; //array of vals used in calulations
    float* oldAry; //array that stores old plate vals
    int* interpolatedAry; //array for interpolated values

    /*variables to keep track of arys first and 
    last rows (not rows given by other processes)*/
    int firstRow, lastRow;

    //variables for printing results to ppm file
    FILE *outFile;
    char fileName[] = "JacobiResults.ppm";

    //MPI initialization
    MPI_Init(&argc, &argv);
    //determines how many processes there are
    MPI_Comm_size(MPI_COMM_WORLD, &commSize);
    //determines who the current process is
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    //master only code
    if(rank == 0) {

        //start timing
        startTime = MPI_Wtime();
        
        //prints normal header information to console
        printf("Isabella Patnode ~ COMP233.A ~ Hybrid Jacobi Iteration for Heat Distribution\n");
        printf("Original code taken from Argonne National Labs\n");
        printf("This code solves a Laplace eq. using Jacobi iterations with Hybrid\n");
        printf("The number of processes is %d\n\n", commSize);

        //opens or creates ppm file
        outFile = fopen(fileName, "w");

        //checks if file was open/created
        if(outFile == NULL) {
            printf("Error in creating/opening file\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        //prints type of ppm version we want to use to the file
        fprintf(outFile, "P3\n");
        //prints dimensions of the image to the file
        fprintf(outFile, "%d %d #image width (cols) and height (rows)\n", MAX_DIM, MAX_DIM);
        //prints header information to the file
        fprintf(outFile, "# Isabella Patnode ~ COMP233.A ~ Hybrid Laplace Heat Distribution\n");
        //prints max pixel size to file
        fprintf(outFile, "255 #max pixel size\n");

    } //end of master only code

    //sets number of threads to be used
    omp_set_num_threads(NUM_THREADS);

    //calculates the starting size of each array
    arySize = MAX_DIM / commSize;

    //calculates the remaining rows
    remainderRows = MAX_DIM % commSize;

    if(remainderRows != 0) {
        if(rank == 0 || rank == commSize - 1) {
            arySize += remainderRows / 2;
        }
    }

    /*creates dynamically allocated arrays where the array of 
    MAX_DIM is split up into ind. arrays between processes*/
    currentAry = (float*)malloc((arySize + EXTRA_ROWS) * MAX_DIM * sizeof(float));
    oldAry = (float*)malloc((arySize + EXTRA_ROWS) * MAX_DIM * sizeof(float));

    interpolatedAry = (int*)malloc(arySize * MAX_DIM * sizeof(int));
    
    firstRow = 1;
    lastRow = arySize;

    //first and last processes have one less interior row
    if(rank == 0) {
        firstRow++;
    }
    if(rank == commSize - 1) {
        lastRow--;
    }

    //initializes exterior rows
    for(c = 0; c < MAX_DIM; c++) {
        *(currentAry + (firstRow - 1) * MAX_DIM + c) = NORTH;
        *(currentAry + (lastRow + 1) * MAX_DIM + c) = SOUTH;

        *(oldAry + (firstRow - 1) * MAX_DIM + c) = NORTH;
        *(oldAry + (lastRow + 1) * MAX_DIM + c) = SOUTH;
    }

    //initializes interior rows
    for(r = firstRow; r <= lastRow; r++) {
        //initializes exterior cells of interior rows to bound. vals.
        *(currentAry + r * MAX_DIM + 0) = WEST;
        *(currentAry + r * MAX_DIM + (MAX_DIM - 1)) = EAST;

        *(oldAry + r * MAX_DIM + 0) = WEST;
        *(oldAry + r * MAX_DIM + (MAX_DIM - 1)) = EAST;

        //initializes interior cells to sum of boundary values
        for(c = 1; c < MAX_DIM - 1; c++) {
            *(currentAry + r * MAX_DIM + c) = (NORTH + SOUTH + EAST + WEST) / NGH_CELL;
            *(oldAry + r * MAX_DIM + c) = (NORTH + SOUTH + EAST + WEST) / NGH_CELL;
        }
    }

    iterCount = 0;

    do{
        //sends bottom row in ary to next process if the process is not the last process
        if(rank < commSize - 1) {
            MPI_Send(currentAry + (arySize * MAX_DIM), MAX_DIM, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD);
        }

        //process receives its "top" row from the process before it if process is not the first process
        if(rank > 0) {
            MPI_Recv((currentAry + 0 * MAX_DIM), MAX_DIM, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        //sends top row in ary to process before it if the process is not the first process
        if(rank > 0) {
            MPI_Send((currentAry + 1 * MAX_DIM), MAX_DIM, MPI_FLOAT, rank - 1, 1, MPI_COMM_WORLD);
        }

        //process receives its "bottom" row from the process after it if process is not the last process
        if(rank < commSize - 1) {
            MPI_Recv((currentAry + (arySize + 1) * MAX_DIM), MAX_DIM, MPI_FLOAT, rank + 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

        iterCount++;

        diffNorm = 0.0;

#pragma omp parallel for shared(firstRow, lastRow, MAX_DIM, NGH_CELL), private(r, c), reduction(+: diffNorm)
        for(r = firstRow; r <= lastRow; r++) {
            for(c = 1; c < MAX_DIM - 1; c++) {
                //calculates the next temp of each internal cell
                *(oldAry + r * MAX_DIM + c) = (*(currentAry + (r - 1) * MAX_DIM + c) + 
                        *(currentAry + (r + 1) * MAX_DIM + c) + *(currentAry + r * MAX_DIM + (c - 1))
                         + *(currentAry + r * MAX_DIM + (c + 1))) / NGH_CELL;

                //sums the squares of the internal cells
                diffNorm += (*(oldAry + r * MAX_DIM + c) - *(currentAry + r * MAX_DIM + c)) 
                        * (*(oldAry + r * MAX_DIM + c) - *(currentAry + r * MAX_DIM + c));
            }
        } //end of parallel section

        //sets current array values as new temps and old ary values as prev temps
        temp = currentAry;
        currentAry = oldAry;
        oldAry = temp;

        //sends sum of diffNorm to all processes
        MPI_Allreduce(&diffNorm, &gDiffNorm, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

        gDiffNorm = sqrt(gDiffNorm);

        //master only code
        if(rank == 0) {
            //prints diffNorm value and # of iterations to console every 1000 iterations
            if((iterCount % PRINT_ITER) == 0) {
                printf("At iteration %d, the value of diffNorm is %f\n", iterCount, gDiffNorm);
            }
        } //end of master only code

    /* continues calculating Jacobi approximations until 
    ** the iteration converges or 500000 iterations are completed */
    } while(gDiffNorm > EPSILON && iterCount < MAX_ITER); //end of do-while

    //green is always zero for each cell's RGB value
    green = 0;

    //each process interpolates their final values
    for(r = 1; r <= arySize; r++) {
        for(c = 0; c < MAX_DIM; c++) {
            //interpolate values
            red = (*(currentAry + r * MAX_DIM + c) * MAX_RGB) / NORTH;
            blue = MAX_RGB - red;

            //combines RGB values back into one value
            *(interpolatedAry + (r - 1) * MAX_DIM + c) = (red << 16) + (green << 8) + blue;           
        }
    }

    //each process sends their final arrays to master
    if(rank != 0) {
        MPI_Send(interpolatedAry, arySize * MAX_DIM, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    //master only code
    else {
        fprintf(outFile, "# The image took %d iterations to converge\n", iterCount);

        numFileCols = 0;

        recvSize = arySize;

        for(proc = 0; proc < commSize; proc++) {
            //receives array from processes except itself (master)
            if(proc != 0 && proc != commSize - 1) {
                //processes that are not the first and last process have either 2 or 4 less rows so size of the array being received is smaller
                MPI_Recv(interpolatedAry, (arySize - (remainderRows / 2)) * MAX_DIM, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                //saves the size of the array received
                recvSize = arySize - (remainderRows / 2);
            }
            else if(proc == commSize - 1) {
                MPI_Recv(interpolatedAry, arySize * MAX_DIM, MPI_INT, proc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                recvSize = arySize;
            }

            numFileCols = 0;

            //prints final array results to ppm file
            for(r = 0; r < recvSize; r++) {
                for(c = 0; c < MAX_DIM; c++) {
                    numFileCols++;

                    rgb = *(interpolatedAry + r * MAX_DIM + c);

                    //extracts each RGB value
                    red = (rgb & 0x00FF0000) >> 16;
                    green = (rgb & 0x0000FF00) >> 8;
                    blue = (rgb & 0x000000FF);
                    
                    fprintf(outFile, "%d %d %d ", red, green, blue);

                    //only prints 15 values (5 tuples) in one row
                    if(numFileCols == FILE_COLS){
                        fprintf(outFile, "\n");
                        numFileCols = 0;
                    }
                }
            }
        } //end of receiving and printing each proc's array to ppm file
    } //end of master only code


    //garbage collection
    free(currentAry);
    free(oldAry);
    free(interpolatedAry);

    MPI_Finalize();

    //master only code
    if(rank == 0) {
        fclose(outFile);

        stopTime = MPI_Wtime();

        printf("The runtime is %f seconds\n", stopTime - startTime);
        
        printf("Normal termination\n");
    }//end of master only code

    return 0;
}