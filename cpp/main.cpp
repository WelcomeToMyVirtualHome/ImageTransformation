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
    ga->SetOperators(GeneticAlgorithm::CrossoverFlags::CYCLE, GeneticAlgorithm::MutationFlags::SINGLE_SWAP);
    int i = 0;
    int iMax = 1500;


    while(true)
    {
        ga->Fitness();
        printf("i=%d, AVG fit=%.6f\n",i,ga->AverageFitness());
        if(i++ == iMax)
            break;
        
        ga->NewGeneration(ga->SelectParents(200,5,i,true));  
    }
    

    return 0;
}