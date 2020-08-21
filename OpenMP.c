/****************************************************************************
*Isabella Patnode ~ COMP233.A ~ OpenMP Jacobi Iteration for Heat Distribution
***********Code originally taken from Argonne National Laboratory************
****************************************************************************/

#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

//values of the boundary cells
const int NORTH = 100;
const int SOUTH = 100;
const int WEST = 0;
const int EAST = 0;

const int MAX_DIM = 1000; //dimensions of ary
const int MAX_ITER = 500000; //max num of iterations

//value approx. must be less than to converge
const float EPSILON = 1.0e-2;
const int NGH_CELL = 4; //num of neighboring cells

const int MAX_RGB = 255; //max RGB value

/*variable used to calculate when to 
**print current iter. and diffNorm */
const int PRINT_ITER = 1000;
const int FILE_COLS = 15;

int main(int argc, char* argv[]) {
    int r, c; //loop variables
    int iterCount; //tracks number of iterations
    float diffNorm; //variables for Jac. calc.
    float* temp; //variable for array swapping
    float startTime, stopTime; //variables for timing
    int numFileCols; //tracks the number of values in one row the file
    float* currentAry; //array of values used in calculations
    float* oldAry; //array that stores old plate values
    int numThreads;
    int red, blue, green; //values for interpolation
    char* cmdEnd;

    //variables for printing results to ppm file
    FILE *outFile;
    char fileName[] = "JacobiResults.ppm";

    //starts timing
    startTime = omp_get_wtime();

    
    //checks that number of threads has been passed from cmd line
    if(argc < 2) {
        printf("Error: too few arguments\n");
        exit(1);
    }

    //gets number of threads from cmd line
    numThreads = strtol(argv[1], &cmdEnd, 10);

    printf("The number of threads are %d", numThreads);

    //sets number of threads to be used
    omp_set_num_threads(numThreads);

    //Prints out header information to console
    printf("Isabella Patnode ~ COMP233.A ~ OpenMP Jacobi Iteration for Heat Distribution\n");
    printf("Original code taken from Argonne National Labs\n");
    printf("This code solves a Laplace equation using Jacobi iterations with OpenMP\n\n");

    //opens or creates ppm file
    outFile = fopen(fileName, "w");

    //checks if file was open/created
    if(outFile == NULL) {
        printf("Error in creating/opening file\n");
        exit(1);
    }

    //prints type of ppm version we want to use to the file
    fprintf(outFile, "P3\n");
    //prints dimensions of the image to the file
    fprintf(outFile, "%d %d #image width (cols) and height (rows)\n", MAX_DIM, MAX_DIM);
    //prints header information to the file
    fprintf(outFile, "# Isabella Patnode ~ COMP233.A ~ Laplace Heat Distribution\n");
    //prints max pixel size to file
    fprintf(outFile, "255 #max pixel size\n");
    
    //creates arrays dynamically with each having one block of memory
    currentAry = (float*)malloc(MAX_DIM * MAX_DIM * sizeof(float));
    oldAry = (float*)malloc(MAX_DIM * MAX_DIM * sizeof(float));

    //initializes exterior rows to bound. vals.
    for(c = 0; c < MAX_DIM; c++) {
        *(currentAry + 0 * MAX_DIM + c) = NORTH;
        *(currentAry + (MAX_DIM - 1) * MAX_DIM + c) = SOUTH;

        *(oldAry + 0 * MAX_DIM + c) = NORTH;
        *(oldAry + (MAX_DIM - 1) * MAX_DIM + c) = SOUTH;
    }

    //initializes interior rows
    for(r = 1; r < MAX_DIM - 1; r++) {
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

    do {
        iterCount++;

        diffNorm = 0.0;

#pragma omp parallel for reduction(+: diffNorm) private(r, c)
        for(r = 1; r < MAX_DIM - 1; r++) {
            for(c = 1; c < MAX_DIM - 1; c++) {
                //calculates the next temp of each internal cell
                *(oldAry + r * MAX_DIM + c) = (*(currentAry + (r - 1) * MAX_DIM + c) + 
                        *(currentAry + (r + 1) * MAX_DIM + c) + *(currentAry + r * MAX_DIM + (c - 1))
                         + *(currentAry + r * MAX_DIM + (c + 1))) / (float) NGH_CELL;
                
                //sums the squares of the internal cells
                diffNorm += (*(oldAry + r * MAX_DIM + c) - *(currentAry + r * MAX_DIM + c)) 
                        * (*(oldAry + r * MAX_DIM + c) - *(currentAry + r * MAX_DIM + c));
            }
        } //end of parallel section

        //sets current array values as new temps and old ary values as prev temps
        temp = currentAry;
        currentAry = oldAry;
        oldAry = temp;

        //prints diffNorm value and # of iterations to console every 1000 iterations
        if((iterCount % PRINT_ITER) == 0) {
            printf("At iteration %d, the value of diffNorm is %f\n", iterCount, diffNorm);
        }

        diffNorm = sqrt(diffNorm);

    /* continues calculating Jacobi approximations until 
    ** the iteration converges or 500000 iterations are completed */
    } while(diffNorm > EPSILON && iterCount < MAX_ITER); //end of do-while

    fprintf(outFile, "# The image took %d iterations to converge\n", iterCount);

    numFileCols = 0;

    //prints final array results to ppm file
    for(r = 0; r < MAX_DIM; r++) {
        for(c = 0; c < MAX_DIM; c++) {
            numFileCols++;

            //interpolate values
            red = (*(currentAry + r * MAX_DIM + c) / NORTH) * MAX_RGB;
            green = 0;
            blue = MAX_RGB - red;
            
            fprintf(outFile, "%d %d %d", red, green, blue);

            //only prints 15 values (5 tuples) in one row
            if(numFileCols == FILE_COLS){
                fprintf(outFile, "\n");
                numFileCols = 0;
            }
        }
    }

    //garbage collection
    free(currentAry);
    free(oldAry);
    fclose(outFile);

    //stops time
    stopTime = omp_get_wtime();

    //prints runtime to console
    printf("The runtime is %f seconds\n", stopTime - startTime);

    printf("Normal termination\n");

    return 0;
}