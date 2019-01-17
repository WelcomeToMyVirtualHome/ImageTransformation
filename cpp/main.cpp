#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <dirent.h>
#include <fstream>
#include <cstdlib>
#include <csignal>
#include "image.h"
#include "geneticAlgorithm.h"
#include "resources.h"

volatile sig_atomic_t flag = 0;
void my_function(int sig)
{
    flag = 1;
}

int main(int argc, char** argv)
{
    signal(SIGINT, my_function); 

    if(argc < 4)
    {
        std::cout <<"Usage: input_folder output_folder params_file" << "\n";
        return -1;
    }
    std::cout << "OpenCV " << CV_VERSION << "\n";
    
    Resources *res = new Resources(argc,argv);    
    GeneticAlgorithm *ga = new GeneticAlgorithm(res);

    int generationSize = 1000;
    int i = 0;
    int iMax = 500;
    
    ga->CreateGeneration(generationSize);
    ga->SetOperators(GeneticAlgorithm::CrossoverFlags::CYCLE, GeneticAlgorithm::MutationFlags::SINGLE_SWAP, GeneticAlgorithm::GoalFunctionFlags::MSSIM);
    
    while(true)
    {
        ga->Fitness();
        ga->writeToFile(i);
        ga->writeImages(i,20,true);
        ga->NewGeneration(ga->SelectParents(200,5,i),0.08);  

        if(flag)
        {
            printf("\nSignal caught!\nFlushing data\n");
            ga->Flush();
            return 0;
        }     

        if(++i == iMax)
            break;
    }
    
    return 0;
}