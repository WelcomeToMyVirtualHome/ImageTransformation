#pragma once 

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Resources
{
public:
	Resources(int argc, char** argv)
	{
		Load(argc,argv);
		CreateLattice();
	}

	int Load(int argc, char **argv)
	{
		printf("Loading...\n");
	    char buffer[50];
	    sprintf(buffer, "%s/%s", argv[1], INPUT_RESIZED);
	    image = cv::imread(buffer, cv::IMREAD_UNCHANGED);

	    if(!image.data)
	    {
	        std::cout <<  "Could not open or find the image" << "\n";
	        return -1;
	    }
	    if(image.cols != image.rows){
	        std::cout << "Image not square" << "\n";
	        return 0;
	    }
	    cv::imshow("Image",image);
	    cv::waitKey(0);

	    outputPath = argv[2];

	    DIR *dir;
	    struct dirent *ent;
	    printf("Reading from %s\n", argv[1]);
	    if((dir = opendir(argv[1])) != NULL)
	    {
	        while((ent = readdir(dir)) != NULL)
	        {
	            printf("%s\n", ent->d_name);
	            sprintf(buffer, "%s/%s", argv[1], ent->d_name);
	            cv::Mat img = cv::imread(buffer, cv::IMREAD_UNCHANGED);
	            if(img.data && std::string(ent->d_name).compare(std::string(INPUT_RESIZED)) != 0)
	                extracted.add(++nImages,img);
	            else if(!img.data)
	            	printf("No data in %s\n", ent->d_name);
	        }
	        closedir(dir);
	    } 
	    else 
	    {
	        perror("LoadImages: Empty directory");
	        return EXIT_FAILURE;
	    }

	    std::string line;
	    std::ifstream paramsStream;
	    paramsStream.open(argv[3]);
	    if (paramsStream.is_open())
	    {
	        while(std::getline(paramsStream,line))
	        {
	            int p = atoi(line.c_str());
	            params.push_back(p);
	        }
	        paramsStream.close();
	    }

	    imageSize = params[0];
	    latticeN = params[1];
	    latticeConst = std::floor(float(imageSize)/latticeN);
	    printf("Done...\n");
	    return 0;
	}

	void CreateLattice()
	{
		printf("Create lattice...\n");
	    lattice.reserve(latticeN*latticeN);
	    for(int i = 0; i < latticeN; i++)
	        for(int j = 0; j < latticeN; j++)
	            lattice.push_back(std::pair<int,int>(i*latticeConst,j*latticeConst));
	    printf("...Done\n");
	}
	
	char *outputPath;

	cv::Mat image;
	Image extracted;
	
	std::vector<std::pair<int,int> >lattice;
	int latticeConst = 0;
	int latticeN = 0;
	int imageSize = 0;
	int nImages = 0;
private:
	std::vector<int> params;
	const char* INPUT_RESIZED = "input_resized.png";
};