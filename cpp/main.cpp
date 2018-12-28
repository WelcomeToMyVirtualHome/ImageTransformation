#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include <dirent.h>
#include <fstream>

using namespace cv;

const char* INPUT_RESIZED = "input_resized.png";
std::vector<Mat> img;
std::vector<int> params;
Mat image;

std::vector<std::vector<Mat> > generation;
std::pair<int,int> *lattice;
int lattice_const;

void CreateLattice()
{
    lattice_const = params[0]/params[1];
    lattice = new std::pair<int,int>[params[1]*params[1]];
    for(int i = 0; i < params[1]; i++)
        for(int j = 0; j < params[1]; j++)
            lattice[i+j] = std::pair<int,int>(i*lattice_const,j*lattice_const);
}

int Load(int argc, char **argv)
{
    char buffer[50];
    sprintf(buffer, "%s/%s", argv[1], INPUT_RESIZED);
    image = imread(buffer, CV_LOAD_IMAGE_COLOR);

    if(!image.data)
    {
        std::cout <<  "Could not open or find the image" << "\n";
        return -1;
    }

    DIR *dir;
    struct dirent *ent;
    printf("Reading from %s\n", argv[1]);
    if((dir = opendir(argv[1])) != NULL)
    {
        while((ent = readdir(dir)) != NULL)
        {
            printf("%s\n", ent->d_name);
            sprintf(buffer, "%s/%s", argv[1], ent->d_name);
            image = imread(buffer, CV_LOAD_IMAGE_COLOR);
            if(image.data && std::string(ent->d_name).compare(std::string(INPUT_RESIZED)) != 0)
            {
                img.push_back(image);
            }
        }
        closedir(dir);
    } 
    else 
    {
        perror("error");
        return EXIT_FAILURE;
    }
    return 0;
}

void ReadParams(char* params_file)
{
    std::string line;
    std::ifstream params_stream;
    params_stream.open(params_file);
    if (params_stream.is_open())
    {
        while(std::getline(params_stream,line))
        {
            int p = atoi(line.c_str());
            params.push_back(p);
        }
        params_stream.close();
    }
}

void DrawImage(size, images)
{

}

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        std::cout <<" Usage: input_folder output_folder params_file" << "\n";
        return -1;
    }
    Load(argc, argv);
    ReadParams(argv[3]);
    CreateLattice();

    return 0;
}