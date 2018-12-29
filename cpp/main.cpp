#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <string>
#include <dirent.h>
#include <fstream>
#include <assert.h> 

using namespace cv;

const char* INPUT_RESIZED = "input_resized.png";
std::vector<Mat> extracted;
std::vector<int> params;
Mat image;

std::vector<std::vector<Mat> > generation;
std::pair<int,int> *lattice;
int lattice_const;

void CreateLattice()
{
    lattice_const = params[0]/params[1];
    int size = params[1];
    lattice = new std::pair<int,int>[size*size];
    for(size_t i = 0; i < size; i++)
        for(size_t j = 0; j < size; j++)
            lattice[i*size+j] = std::pair<int,int>(i*lattice_const,j*lattice_const);
        
}

int Load(int argc, char **argv)
{
    char buffer[50];
    sprintf(buffer, "%s/%s", argv[1], INPUT_RESIZED);
    image = imread(buffer, -1);

    if(!image.data)
    {
        std::cout <<  "Could not open or find the image" << "\n";
        return -1;
    }
    imshow("image",image);
    waitKey(0);
    DIR *dir;
    struct dirent *ent;
    printf("Reading from %s\n", argv[1]);
    if((dir = opendir(argv[1])) != NULL)
    {
        while((ent = readdir(dir)) != NULL)
        {
            printf("%s\n", ent->d_name);
            sprintf(buffer, "%s/%s", argv[1], ent->d_name);
            Mat img = imread(buffer, -1);
            if(img.data && std::string(ent->d_name).compare(std::string(INPUT_RESIZED)) != 0)
            {
                extracted.push_back(img);
            }
        }
        closedir(dir);
    } 
    else 
    {
        perror("LoadImages: Empty directory");
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

void DrawImage(std::vector<Mat> images)
{
    Mat output(image.rows, image.cols, image.type());
    size_t len = extracted.size() * extracted.size();
    for(int i = 0; i < len; i++)
        if(output.type() == extracted[i].type() && extracted[i].rows <= output.rows and extracted[i].cols <= output.cols)
            extracted[i].copyTo(output(Rect(lattice[i].first, lattice[i].second,lattice_const,lattice_const)));
    imshow("img",output);
    waitKey(0);

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
    DrawImage(extracted);

    return 0;
}