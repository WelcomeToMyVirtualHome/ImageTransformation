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

float avg(std::vector<float> &v)
{
    float return_value = 0.0;
    int n = v.size();
   
    for ( int i=0; i < n; i++)
    {
        return_value += v[i];
    }
   
    return ( return_value / ((float)n));
}

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        std::cout <<" Usage: input_folder output_folder params_file" << "\n";
        return -1;
    }

    Resources *res = new Resources(argc,argv);    
  
    GeneticAlgorithm ga(res);
    ga.CreateGeneration(10);
  
    ga.Fitness();   
    for(auto img : ga.getGeneration())
        printf("Fitness=%f\n", img.getFitness());
    
    std::vector<Image> parents = ga.SelectParents(1,2,0);
    printf("Parents\n");
    for(auto p : parents)
    {
        p.Show(0);
        printf("Fitness=%f\n", p.getFitness());
    }
    return 0;
}