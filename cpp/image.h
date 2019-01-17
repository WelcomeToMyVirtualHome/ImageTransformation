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
	std::array<bool,3> rotation = { {0,0,0} };
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
	}
		
	void add(int index, cv::Mat img)
	{
		Data data;
		data.index = index;
		data.img = img;
		data.rotation = { {0,0,0} };
		images.push_back(data); 
	}

	void shuffle(bool all = true, int begin = 0, int end = 0) 
	{ 
		all ? std::random_shuffle(images.begin(), images.end()) : std::random_shuffle(images.begin() + begin, images.begin() + end);
	}
	
	void put(const std::vector<std::pair<int,int> > &lattice, int lattice_const, bool modify = false)
	{
		if(lattice.size() != images.size())
			printf("Wrong sizes\n");

		std::vector<Data> copy(images.size());
		for(uint i = 0; i < images.size(); i++)
		{
			Data data;
			data.index = images[i].index;
			data.img = images[i].img.clone();
			data.rotation = images[i].rotation;
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
		for(auto &img : images)
		{
			for(uint i = 0; i < img.rotation.size(); i++)
				if(img.rotation[i])
					RotateClockwise(img);
		}
	}

	void RotateClockwise(Data &image)
	{
		cv::transpose(image.img, image.img);	
		cv::flip(image.img, image.img,cv::RotateFlags::ROTATE_90_CLOCKWISE);
	}

	void ScaleChannel(int n, int multiplier = 1, int channel = 0)
	{
		for(int i = 0; i < images[n].img.rows; i++)
		{
			for(int j = 0; j < images[n].img.cols; j++)
			{
				auto pixel = images[n].img.at<cv::Vec4b>(i, j);
				float pixelValue = (float)pixel[channel];
				pixelValue *= multiplier;
				pixel[channel] = uchar(pixelValue);
				images[n].img.at<cv::Vec4b>(i, j) = pixel;
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

private:
	std::vector<Data> images;
	cv::Mat image;
	double fitness = 0;
};