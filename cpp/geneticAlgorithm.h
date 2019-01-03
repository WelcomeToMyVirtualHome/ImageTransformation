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

	float fitness_func(float score, int i){
		return score*((i+1.0)/1000.0) + 1.0;
	}

	std::tuple<std::vector<float>,std::vector<float>> Fitness(std::vector<Image> generation, int i, const cv::Mat &image) 
	{
		std::vector<float> n_fitness(generation.size(), 0.0);
		std::vector<float> fitness(generation.size(), 0.0);
		float score = 0.0;
		for (uint g = 0; g < generation.size(); ++g)
		{
			//size = 5, has to be changed
			score = MSE(generation[g].getImage(), generation[g].getImage(), image, 5); 
   	      	n_fitness[g] = fitness_func(score,i);
   			fitness[g] = score;
		}
		return {fitness, n_fitness};
	}

	//TEST
	std::vector<Image> selectParents(int nParents, std::vector<float> n_fitness )
	{	
		std::vector<Image> parents;
		for (int i = 0; i < nParents; ++i)
			parents.push_back(generation[weighted_random_choice(n_fitness)]);
		return parents;
	}

	int weighted_random_choice(std::vector<float> n_fitness)
	{
		float max = std::accumulate(n_fitness.begin(), n_fitness.end(), 0.0f);
		std::random_device rd;     // only used once to initialise (seed) engine
		std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
		std::uniform_real_distribution<float> uni(0,max); // guaranteed unbiased
		float pick = uni(rng);
		float current = 0;
		int size = n_fitness.size();
		for (int i = 0; i < size; ++i)
		{
			current += n_fitness[i];
			if(current > pick)
				return i;
		}
	}

	template <typename T>
	std::vector<size_t> sort_indexes(const std::vector<T> &v) {

	  // initialize original index locations
	  std::vector<size_t> idx(v.size());
	  std::iota(idx.begin(), idx.end(), 0);

	  // sort indexes based on comparing values in v
	  sort(idx.begin(), idx.end(),
	       [&v](size_t i1, size_t i2) {return v[i1] > v[i2];}); /* < or > */

	  return idx;
	}

	//TEST
	std::vector<Image> selectBest(std::vector<float> fitness, int n_best)
	{
		std::vector<size_t> bestIndexes = sort_indexes(fitness);
		std::vector<Image> best;
		for (int i = 0; i < n_best; ++i)
			best.push_back(generation[bestIndexes[i]]);
		return best;
	}

	void newGeneration(std::vector<float> &fitness, int n_parents, int n_best)
	{
		
	}

	const std::vector<Image> &getGeneration()
	{
		return generation;
	}

private:
	std::vector<Image> generation;
};