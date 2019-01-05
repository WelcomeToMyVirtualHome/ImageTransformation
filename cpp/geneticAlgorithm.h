#pragma once

#include "image.h"
#include "resources.h"

class GeneticAlgorithm
{
public:
	GeneticAlgorithm(Resources *res) 
	{
		this->res = res;
		srand48(time(NULL));
	} 

	~GeneticAlgorithm()
	{

	}

	void CreateGeneration(int generationSize)
	{
	    this->generationSize = generationSize;
		generation.resize(generationSize);
	    for(int i = 0; i < generationSize; i++)
	    {
	    	Image newImage(res->image, res->extracted);
	        newImage.shuffle();
	        newImage.put(res->lattice, res->lattice_const);
	        generation[i] = newImage;
	    }
	}

	float MSE(const cv::Mat &imageToCompare)
	{
		float mse = 0;
		for(int i = 0; i < res->lattice_n; i++)
		{
			for(int j = 0; j < res->lattice_n; j++)
			{
				auto m1 = cv::mean(imageToCompare(cv::Rect(i*res->lattice_const, j*res->lattice_const, res->lattice_const, res->lattice_const)));
				auto m2 = cv::mean(res->image(cv::Rect(i*res->lattice_const, j*res->lattice_const, res->lattice_const, res->lattice_const)));
				mse += (m1[0] + m1[1] + m1[2] - m2[0] - m2[1] - m2[2])*(m1[0] + m1[1] + m1[2] - m2[0] - m2[1] - m2[2]);
			}
		}	
		return mse/res->lattice_n/res->lattice_n;
	}

	float FitnessFunc(float score, int i)
	{
		return score + 1.001*i;
	}

	void Fitness() 
	{
		for(auto it = begin(generation); it != end(generation); ++it)
       	  	it->setFitness(1./MSE(it->getImage()));
	}

	float AverageFitness()
	{
		float sum = 0;
		for(auto img : generation)
			sum += img.getFitness();
		return sum/generationSize;
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

	std::vector<Image> SelectParents(int nSelect, int nBest, int iter, bool showBest = false)
	{	
		std::vector<Image> parents(nSelect + nBest);
		std::vector<float> n_fitness(generation.size());
		for(uint i = 0; i < generation.size(); i++)
			n_fitness[i] = FitnessFunc(generation[i].getFitness(),iter);

		for(int i = 0; i < nSelect; ++i)
			parents[i] = generation[WeightedRandomChoice(n_fitness)];

		std::vector<size_t> bestIndexes = SortIndexes(n_fitness);
		for(int i = 0; i < nBest; i++)
			parents[nSelect + i] = generation[bestIndexes[i]];

		if(showBest)
			parents[nSelect+nBest-1].Show(1);
		return parents;
	}

	void NewGeneration(std::vector<Image> parents)
	{
		std::vector<Image> newGeneration(generationSize);
		int nParents = parents.size();
		uint iter = 0;
		while(true)
		{
			if(iter == generationSize)
				break;

			int parent1Index = int(drand48()*nParents);
			int parent2Index = int(drand48()*nParents);
			while(parent1Index == parent2Index)
				parent2Index = int(drand48()*nParents);

			Image parent1 = parents[parent1Index];
			Image parent2 = parents[parent2Index];
			
			Image child(res->image,res->extracted);
			Order1Crossover(child,parent1,parent2);

			if(drand48() < 0.05)
				SingleSwapMutation(child);
			
			child.put(res->lattice, res->lattice_const);

			newGeneration[iter] = child;
			iter++;
		}
		generation = newGeneration;
	}

	void Order1Crossover(Image &child, const Image &parent1, const Image &parent2)
	{
		int nImages = res->extracted.getImages().size();
	
		int crosspoint1 = int(drand48()*nImages);
		int crosspoint2 = int(drand48()*nImages);
		while(crosspoint1 == crosspoint2)
			crosspoint2 = int(drand48()*nImages);

		std::vector<std::pair<int,cv::Mat> > images(nImages);
		int min = std::min(crosspoint1,crosspoint2);
		int max = std::max(crosspoint1,crosspoint2);
		std::vector<std::pair<int,cv::Mat> > parent1Images = parent1.getImages();
		std::vector<std::pair<int,cv::Mat> > parent2Images = parent2.getImages();

		std::vector<std::pair<int,cv::Mat> > fromParent1(max-min+1);
		for(int i = 0; i < max - min + 1; i++)
		{
			images[i+min] = parent1Images[i];
			fromParent1[i] = parent1Images[i];
		}

		RemoveDuplicates(parent2Images,fromParent1);
		SortImages(parent2Images);
		
		for(int i = 0; i < min; i++)
			images[i] = parent2Images[i];	
		
		for(uint i = 0; i < parent2Images.size()-min; i++)
			images[i+max+1] = parent2Images[i+min];
	
		child.setImages(images);
	}

	void SingleSwapMutation(Image &child)
	{
		int nImages = res->extracted.getImages().size();
		
		int index1 = int(drand48()*nImages);
		int index2 = int(drand48()*nImages);
		while(index1 == index2)
			index2 = int(drand48()*nImages);

		std::vector<std::pair<int,cv::Mat> > images = child.getImages();
		auto var = images[index1];
		images[index1] = images[index2];
		images[index2] = var;
		child.setImages(images);
	}

	void SortImages(std::vector<std::pair<int,cv::Mat> > &toSort) 
	{
		sort(std::begin(toSort), std::end(toSort),[&](const auto &lhs, const auto &rhs) {return lhs.first < rhs.first;} );
	}

	void RemoveDuplicates(std::vector<std::pair<int,cv::Mat> > &images, const std::vector<std::pair<int,cv::Mat> > &toRemove)
	{
		for(auto r : toRemove)
			for (auto it = images.begin(); it != images.end(); )
				if(r.first == it->first)
					images.erase(it);
				else 
					++it;
	}
	
	const std::vector<Image> &getGeneration() const
	{
		return generation;
	}

	std::vector<Image> &getGeneration()
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
	uint generationSize = 0;
};