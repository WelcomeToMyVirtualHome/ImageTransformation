#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <numeric>
#include <random>

class Image
{
public:
	Image() 
	{

	}
	
	Image(cv::Mat p_image) 
	{ 
		image = p_image.clone();
	}
	
	Image(cv::Mat p_image, const Image &img)
	{ 
		image = p_image.clone();
		images = img.images;
	}
	
	void add(int index, cv::Mat& img)
	{
		images.push_back(std::pair<int,cv::Mat>(index,img)); 
	}

	void shuffle() 
	{ 
		std::random_shuffle(images.begin(), images.end()); 
	}
	
	void put(const std::vector<std::pair<int,int> > &lattice, int lattice_const, bool show=false, int wait_ms = 0)
	{
		if(lattice.size() != images.size())
			printf("Wrong sizes\n");

		for(uint i = 0; i < images.size(); i++)
	        if(image.type() == images[i].second.type() && images[i].second.rows <= image.rows and images[i].second.cols <= image.cols)
	            images[i].second.copyTo(image(cv::Rect(lattice[i].first, lattice[i].second,lattice_const,lattice_const)));   
	    if(show)
	    {
	        Show(wait_ms);
	    }
	}

	void Show(int wait_ms = 0)
	{
	    cv::imshow("img",image);
        cv::waitKey(wait_ms);    
	}

	void printOrder()
	{
		for(auto img : images)
			printf("i=%d\n",img.first);
		printf("\n");
	}

	cv::Mat const &getImage() const
	{ 
		return image; 
	}

	void setFitness(float n_fitness)
	{
		fitness = n_fitness;
	}

	double getFitness()
	{
		return fitness;
	}

	void setImage(int i, std::pair<int,cv::Mat> image)
	{
		images[i] = image;
	}

	void setImages(std::vector<std::pair<int,cv::Mat> > n_images)
	{
		images = n_images;
	}

	std::vector<std::pair<int,cv::Mat> > const &getImages() const
	{ 
		return images; 
	}

private:
	std::vector<std::pair<int,cv::Mat> > images;
	cv::Mat image;
	double fitness = 0;
};