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
    int i = 0;
    int iMax = 500;
    
    ga->CreateGeneration(generationSize);
    ga->SetOperators(GeneticAlgorithm::CrossoverFlags::CYCLE, GeneticAlgorithm::MutationFlags::SINGLE_SWAP);
    
    while(true)
    {
        ga->Fitness();
        ga->writeToFile(i);
        ga->writeImages(i,20,true);
        ga->NewGeneration(ga->SelectParents(200,5,i),0.08);  
        
        if(++i == iMax)
            break;
    }
    
    return 0;
}