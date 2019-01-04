#pragma once

#include "image.h"
#include "resources.h"

class GeneticAlgorithm
{
public:
	GeneticAlgorithm(Resources *res) 
	{
		this->res = res;
	} 

	void CreateGeneration(int n)
	{
	    generation.resize(n);
	    for(int i = 0; i < n; i++)
	    {
	    	Image newImage(res->extracted, res->image);
	        newImage.shuffle();
	        newImage.put(res->lattice, res->lattice_const);
	        generation[i] = newImage;
	     }
	}

	float MSE(const cv::Mat &imageToCompare)
	{
		int n = res->lattice_n;
		int size = std::floor(res->image.cols/n);
		float mse = 0;
		for(int i = 0; i < n; i++)
		{
			for(int j = 0; j < n; j++)
			{
				auto m1 = cv::mean(imageToCompare(cv::Rect(i*size, j*size, size, size)));
				auto m2 = cv::mean(res->image(cv::Rect(i*size, j*size, size, size)));
				mse += (m1[0] + m1[1] + m1[2] - m2[0] - m2[1] - m2[2])*(m1[0] + m1[1] + m1[2] - m2[0] - m2[1] - m2[2]);
			}
		}	
		return mse/n/n;
	}

	float FitnessFunc(float score, int i)
	{
		return score*((i+1.0)/1000.0) + 1.1*i;
	}

	void Fitness() 
	{
		for(auto it = begin(generation); it != end(generation); ++it)
       	  	it->setFitness(MSE(it->getImage()));
	}

	std::vector<Image> SelectParents(int nParents, int nBest, int iter)
	{	
		std::vector<Image> parents(nParents + nBest);
		std::vector<float> n_fitness(generation.size());
		for(uint i = 0; i < generation.size(); i++)
			n_fitness[i] = FitnessFunc(generation[i].getFitness(),iter);

		for(int i = 0; i < nParents; ++i)
			parents[i] = generation[WeightedRandomChoice(n_fitness)];

		std::vector<size_t> bestIndexes = SortIndexes(n_fitness);
		for(int i = 0; i < nBest; i++)
			parents[nParents + i] = generation[bestIndexes[i]];
		
		return parents;
	}

	int WeightedRandomChoice(std::vector<float> n_fitness)
	{
		float max = std::accumulate(n_fitness.begin(), n_fitness.end(), 0.0f);
		std::random_device rd;     // only used once to initialise (seed) engine
		std::mt19937 rng(rd());    // random-number engine used (Mersenne-Twister in this case)
		std::uniform_real_distribution<float> uni(0,max); // guaranteed unbiased
		float pick = uni(rng);
		float current = 0;
		for (uint i = 0; i < n_fitness.size(); ++i)
		{
			current += n_fitness[i];
			if(current > pick)
				return i;
		}
		return 0;
	}

	void NewGeneration(std::vector<float> &fitness, int n_parents, int n_best)
	{
		
	}

	std::vector<Image> getGeneration()
	{
		return generation;
	}

	template <typename T>
	std::vector<size_t> SortIndexes(const std::vector<T> &v) 
	{
	  // initialize original index locations
	  std::vector<size_t> idx(v.size());
	  std::iota(idx.begin(), idx.end(), 0);
	  // sort indexes based on comparing values in v
	  sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v[i1] > v[i2];}); /* < or > */
	  return idx;
	}

private:
	std::vector<Image> generation;
	Resources *res;
};