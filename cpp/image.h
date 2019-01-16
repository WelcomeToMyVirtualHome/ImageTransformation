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

	void shuffle(bool all = true, int begin = 0, int end = 0) 
	{ 
		all ? std::random_shuffle(images.begin(), images.end()) : std::random_shuffle(images.begin() + begin, images.begin() + end);
	}
	
	void put(const std::vector<std::pair<int,int> > &lattice, int lattice_const, bool modify = false)
	{
		if(lattice.size() != images.size())
			printf("Wrong sizes\n");

		std::vector<std::pair<int,cv::Mat> > copy(images.size());
		for(uint i = 0; i < images.size(); i++)
			copy[i] = std::pair<int,cv::Mat>(images[i].first,images[i].second.clone());
		
		if(modify)
			Modify(copy);

		for(uint i = 0; i < images.size(); i++)
	        if(image.type() == copy[i].second.type() && copy[i].second.rows <= image.rows and copy[i].second.cols <= image.cols)
	            copy[i].second.copyTo(image(cv::Rect(lattice[i].first, lattice[i].second,lattice_const,lattice_const)));   
	}


	void Modify(std::vector<std::pair<int,cv::Mat> > &images)
	{
		for(auto &img : images)
		{
			for(uint i = 0; i < rotation.size(); i++)
				if(rotation[i])
					RotateClockwise(img);
		}
	}

	void RotateClockwise(std::pair<int,cv::Mat> &image)
	{
		cv::transpose(image.second, image.second);	
		cv::flip(image.second, image.second,cv::RotateFlags::ROTATE_90_CLOCKWISE);
	}

	void ScaleChannel(int n, int multiplier = 1, int channel = 0)
	{
		for(int i = 0; i < images[n].second.rows; i++)
		{
			for(int j = 0; j < images[n].second.cols; j++)
			{
				auto pixel = images[n].second.at<cv::Vec4b>(i, j);
				float pixelValue = (float)pixel[channel];
				pixelValue *= multiplier;
				pixel[channel] = uchar(pixelValue);
				images[n].second.at<cv::Vec4b>(i, j) = pixel;
			}
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

	std::vector<std::pair<int,cv::Mat> > &getImages()
	{ 
		return images; 
	}

private:
	std::vector<std::pair<int,cv::Mat> > images;
	cv::Mat image;
	double fitness = 0;
	std::array<bool,3> rotation = { {0,0,0} };
};