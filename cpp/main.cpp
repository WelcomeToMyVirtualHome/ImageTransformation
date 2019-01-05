#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <dirent.h>
#include <fstream>
#include "image.h"
#include "geneticAlgorithm.h"
#include "resources.h"

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        std::cout <<" Usage: input_folder output_folder params_file" << "\n";
        return -1;
    }

    Resources *res = new Resources(argc,argv);    
    GeneticAlgorithm *ga = new GeneticAlgorithm(res);

    int generationSize = 1000;
    ga->CreateGeneration(generationSize);
   
    int i = 0;
    int iMax = 1000;
    while(true)
    {
        ga->Fitness();
        printf("i=%d, AVG fit=%.3f\n",i,ga->AverageFitness());
        if(i++ == iMax)
            break;

        ga->NewGeneration(ga->SelectParents(500,10,i));
        
        for(int i = 0; i < generationSize; i++)
            if(i % 100 == 0)
                ga->getGeneration()[i].Show(1);
    }
    

    return 0;
}