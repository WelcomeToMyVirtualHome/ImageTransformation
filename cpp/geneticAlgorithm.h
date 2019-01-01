#pragma once

#include "image.h"

class GeneticAlgorithm
{
public:
	GeneticAlgorithm() {} 

	void CreateGeneration(const Image &extracted, const cv::Mat &image, int n)
	{
	    generation.reserve(n);
	    for(int i = 0; i < n; i++)
	    {
	    	Image newImage(extracted, image);
	        newImage.shuffle();
	        generation.push_back(newImage);
	     }
	}

	float MSE(const cv::Mat &img1, const cv::Mat &img2, const cv::Mat &image, int n)
	{
		int size = std::floor(image.cols/n);
		float mse = 0;
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < n; j++)
			{
				cv::Mat sub1 = img1(cv::Rect(i*size, j*size, size, size));
				cv::Mat sub2 = img2(cv::Rect(i*size, j*size, size, size));
				auto m1 = cv::mean(sub1);
				auto m2 = cv::mean(sub2);
				mse += (m1[0] + m1[1] + m1[2] - m2[0] - m2[1] - m2[2])*(m1[0] + m1[1] + m1[2] - m2[0] - m2[1] - m2[2]);
			}
		}
		return mse/n/n;
	}



	// std::map<Image,float> Fitness(std::vector<Image> generation) 
	// std::vector<Image> selectParents() { }
	// std::vector<Image> selectBest() { }

	void newGeneration(std::vector<Image> &generation) {}

	const std::vector<Image> &getGeneration()
	{
		return generation;
	}

private:
	std::vector<Image> generation;
};