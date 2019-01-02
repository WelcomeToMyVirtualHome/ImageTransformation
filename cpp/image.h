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
	
	Image(const Image &img, cv::Mat p_image)
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
	
	std::vector<std::pair<int,cv::Mat> > getImages()
	{ 
		return images; 
	}

	void put(const std::pair<int,int> *lattice, int lattice_const, bool show=false, int wait_ms = 0)
	{
	    for(uint i = 0; i < images.size(); i++)
	        if(image.type() == images[i].second.type() && images[i].second.rows <= image.rows and images[i].second.cols <= image.cols)
	            images[i].second.copyTo(image(cv::Rect(lattice[i].first, lattice[i].second,lattice_const,lattice_const)));   
	    if(show)
	    {
	        cv::imshow("image",image);
	        cv::waitKey(wait_ms);
	    }
	}

	cv::Mat getImage() 
	{ 
		return image; 
	}

private:
	std::vector<std::pair<int,cv::Mat> > images;
	cv::Mat image;
};