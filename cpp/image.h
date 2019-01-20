#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <algorithm>
#include <numeric>
#include <random>

struct Data
{
	int index;
	cv::Mat img;
};

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
		rotation.resize(images.size()*3);
		for(uint i = 0; i < rotation.size(); i++)
			drand48() > 0.5 ? rotation[i] = true : rotation[i] = false;
	}

	Image(const Image &img)
	{
		image = img.image.clone();
		images = img.images;
		rotation = img.rotation;
	}
		
	void add(int index, cv::Mat img)
	{
		Data data;
		data.index = index;
		data.img = img;
		images.push_back(data); 
	}

	void shuffle(bool all = true, int begin = 0, int end = 0) 
	{ 
		all ? std::random_shuffle(images.begin(), images.end()) : std::random_shuffle(images.begin() + begin, images.begin() + end);
	}
	
	void put(const std::vector<std::pair<int,int> > &lattice, int lattice_const, bool modify = true)
	{
		if(lattice.size() != images.size())
			printf("Wrong sizes\n");

		std::vector<Data> copy(images.size());
		for(uint i = 0; i < images.size(); i++)
		{
			Data data;
			data.index = images[i].index;
			data.img = images[i].img.clone();
			copy[i] = data;
		}
		
		if(modify)
			Modify(copy);

		for(uint i = 0; i < images.size(); i++)
	        if(image.type() == copy[i].img.type() && copy[i].img.rows <= image.rows and copy[i].img.cols <= image.cols)
	            copy[i].img.copyTo(image(cv::Rect(lattice[i].first, lattice[i].second,lattice_const,lattice_const)));   
	}


	void Modify(std::vector<Data> &images)
	{
		for(uint i = 0; i < images.size(); i++)
		{
			for(uint k = 0; k < 3; k++)
				if(rotation[i*3 + k])
					RotateClockwise(images[i]);
		}

		for(uint i = 0; i < 3; i++)
			ScaleChannel(bgrShift[i],i);
	}

	void RotateClockwise(Data &image)
	{
		cv::transpose(image.img, image.img);	
		cv::flip(image.img, image.img,cv::RotateFlags::ROTATE_90_CLOCKWISE);
	}

	void ScaleChannel(int add = 0, int channel = 0)
	{
		for(uint img = 0; img < images.size(); img++)
		{
			for(int i = 0; i < images[img].img.rows; i++)
			{
				for(int j = 0; j < images[img].img.cols; j++)
				{
					auto pixel = images[img].img.at<cv::Vec4b>(i, j);
					float pixelValue = (float)pixel[channel];
					pixelValue += add;
					pixel[channel] = uchar(pixelValue);
					images[img].img.at<cv::Vec4b>(i, j) = pixel;
				}
			}
		}
	}
	
	void Show(int wait_ms = 0, std::string name = "img")
	{
	    cv::imshow(name,image);
        cv::waitKey(wait_ms);    
	}

	void printOrder()
	{
		for(auto img : images)
			printf("i=%d\n",img.index);
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

	void setImage(int i, Data image)
	{
		images[i] = image;
	}

	void setImages(std::vector<Data> n_images)
	{
		images = n_images;
	}

	std::vector<Data> const &getImages() const
	{ 
		return images; 
	}

	std::vector<Data> &getImages()
	{ 
		return images; 
	}

	int &getBgrShift(int i)
	{
		return bgrShift[i];
	} 

	std::vector<bool> &getRotation()
	{
		return rotation;
	}

private:
	std::vector<Data> images;
	std::vector<bool> rotation;
	std::array<int,3> bgrShift{ {0,0,0} };
	cv::Mat image;
	double fitness = 0;
};