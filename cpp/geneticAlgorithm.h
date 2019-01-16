#pragma once
#include <thread>
#include <chrono>
#include <cstdio>

#include "image.h"
#include "resources.h"

class GeneticAlgorithm
{
public:

	enum CrossoverFlags {
		CYCLE = 0,
		ORDER_1 = 1,
	};
	
	enum MutationFlags {
	    SINGLE_SWAP = 0,
	    INVERSION = 1,
	    SCRAMBLE = 2, 
	};

	enum GoalFunctionFlags {
	    MSE = 0,
	    MSSIM = 1,
	};

	GeneticAlgorithm(Resources *res) 
	{
		this->res = res;
		char buffer[50];
		sprintf(buffer,"%s/fitnessHistory.dat",res->outputPath);
		output = fopen(buffer,"w");
		srand48(time(NULL));
	} 

	~GeneticAlgorithm()
	{
		fclose(output);
	}

	void CreateGeneration(int generationSize)
	{
	    this->generationSize = generationSize;
		generation.resize(generationSize);
	    for(int i = 0; i < generationSize; i++)
	    {
	    	Image newImage(res->image, res->extracted);
	        newImage.shuffle();
	        newImage.put(res->lattice, res->latticeConst);
	        generation[i] = newImage;
	    }
	    best = generation[0];
	}

	void SetOperators(CrossoverFlags crossoverFlag, MutationFlags mutationFlag, GoalFunctionFlags goalFunctionFlag)
	{
		if(crossoverFlag == CrossoverFlags::ORDER_1)
			crossover = &GeneticAlgorithm::Order1Crossover;
		else if(crossoverFlag == CrossoverFlags::CYCLE)
			crossover = &GeneticAlgorithm::CycleCrossover;
	
		if(mutationFlag == MutationFlags::SINGLE_SWAP)
			mutation = &GeneticAlgorithm::SingleSwapMutation;
		else if(mutationFlag == MutationFlags::INVERSION)
			mutation = &GeneticAlgorithm::InversionMutation;
		else if(mutationFlag == MutationFlags::SCRAMBLE)
			mutation = &GeneticAlgorithm::ScrambleMutation;

		if(goalFunctionFlag == GoalFunctionFlags::MSE)
			goalFunction = &GeneticAlgorithm::getMSE;
		else if(goalFunctionFlag == GoalFunctionFlags::MSSIM)
			goalFunction = &GeneticAlgorithm::getMSSIM;
	}

	void Fitness() 
	{
		for(auto it = begin(generation); it != end(generation); ++it)
			it->setFitness((this->*goalFunction)(it->getImage(), res->image));
	}

	void NewGeneration(std::vector<Image> parents, const float pMutation = 0.05)
	{
		std::vector<Image> newGeneration(generationSize);
		int nParents = parents.size();
		uint iter = 0;
		while(true)
		{
			if(iter >= generationSize)
				break;

			int parent1Index = int(drand48()*nParents);
			int parent2Index = int(drand48()*nParents);
			while(parent1Index == parent2Index)
				parent2Index = int(drand48()*nParents);

			Image parent1 = parents[parent1Index];
			Image parent2 = parents[parent2Index];
			
			Image child1(res->image,res->extracted);
			Image child2(res->image,res->extracted);

			(this->*crossover)(child1,child2,parent1,parent2);

			if(drand48() < pMutation)
				(this->*mutation)(child1);
			if(drand48() < pMutation)
				(this->*mutation)(child2);
			
			child1.put(res->lattice, res->latticeConst, true);
			child2.put(res->lattice, res->latticeConst, true);
			
			newGeneration[iter++] = child1;
			newGeneration[iter++] = child2;
		}
		generation = newGeneration;
	}

	// TEMPORARY
	void ColorMapTest()
	{
		for(int i = 0; i < 3; i++)
		{
			auto image = Image(res->image,res->extracted);
			image.put(res->lattice, res->latticeConst);
			std::cout << "Multiplication " << i << " channel\n";
			for(int k = 0; k < res->nImages; k++)
			{
				image.ScaleChannel(k,2,i);
			}	
			image.put(res->lattice, res->latticeConst);
			image.Show(0);
		}	
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
		best = parents[nSelect+nBest-1];

		return parents;
	}

	void writeToFile(int generation)
	{
		float avg = AverageFitness();
		printf("i=%d, AVG fit=%.6f, best fit=%.6f\n",generation,avg,best.getFitness());	
		fprintf(output,"%d %f %f\n",generation,avg,best.getFitness());
	}

	void writeImages(int iter, int divisor = 1, bool show = false)
	{
		if(show)
			best.Show(1);
		
		if(iter % divisor == 0)
		{
			char buffer[50];
			sprintf(buffer,"%s/best%d.png",res->outputPath,iter);
			cv::imwrite(std::string(buffer),best.getImage());
		}
	}

	const std::vector<Image> &getGeneration() const
	{
		return generation;
	}

	std::vector<Image> &getGeneration()
	{
		return generation;
	}

private:
	std::vector<Image> generation;
	Resources *res;
	uint generationSize = 0;
	void (GeneticAlgorithm::*mutation)(Image&);
	void (GeneticAlgorithm::*crossover)(Image&, Image&, const Image&, const Image&);
	float (GeneticAlgorithm::*goalFunction)(const cv::Mat&, const cv::Mat&);
	FILE *output;
	Image best;

	float getMSSIM( const cv::Mat& i1, const cv::Mat& i2)
	{
		 const double C1 = 6.5025, C2 = 58.5225;
		 /***************************** INITS **********************************/
		 int d= CV_32F;

		 cv::Mat I1, I2;
		 i1.convertTo(I1, d);           // cannot calculate on one byte large values
		 i2.convertTo(I2, d);

		 cv::Mat I2_2   = I2.mul(I2);        // I2^2
		 cv::Mat I1_2   = I1.mul(I1);        // I1^2
		 cv::Mat I1_I2  = I1.mul(I2);        // I1 * I2

		 /***********************PRELIMINARY COMPUTING ******************************/

		 cv::Mat mu1, mu2;   //
		 cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
		 cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

		 cv::Mat mu1_2   =   mu1.mul(mu1);
		 cv::Mat mu2_2   =   mu2.mul(mu2);
		 cv::Mat mu1_mu2 =   mu1.mul(mu2);

		 cv::Mat sigma1_2, sigma2_2, sigma12;

		 GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
		 sigma1_2 -= mu1_2;

		 GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
		 sigma2_2 -= mu2_2;

		 GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
		 sigma12 -= mu1_mu2;

		 ///////////////////////////////// FORMULA ////////////////////////////////
		 cv::Mat t1, t2, t3;

		 t1 = 2 * mu1_mu2 + C1;
		 t2 = 2 * sigma12 + C2;
		 t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

		 t1 = mu1_2 + mu2_2 + C1;
		 t2 = sigma1_2 + sigma2_2 + C2;
		 t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

		 cv::Mat ssim_map;
		 cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

		 cv::Scalar mssim = cv::mean( ssim_map ); // mssim = average of ssim map
		 return mssim[0]+mssim[1]+mssim[2]+mssim[3];
	}
	
	float getMSE(const cv::Mat &imageToCompare, const cv::Mat &image)
	{
		float mse = 0;
		for(int i = 0; i < res->latticeN; i++)
		{
			for(int j = 0; j < res->latticeN; j++)
			{
				auto m1 = cv::mean(imageToCompare(cv::Rect(i*res->latticeConst, j*res->latticeConst, res->latticeConst, res->latticeConst)));
				auto m2 = cv::mean(res->image(cv::Rect(i*res->latticeConst, j*res->latticeConst, res->latticeConst, res->latticeConst)));
				mse += (m1[0] + m1[1] + m1[2] - m2[0] - m2[1] - m2[2])*(m1[0] + m1[1] + m1[2] - m2[0] - m2[1] - m2[2]);
			}
		}	
		return mse/res->latticeN/res->latticeN;
	}

	float FitnessFunc(float score, int i)
	{
		return score/100 + i * 0.002;
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

	void Order1Crossover(Image &child, Image &child2, const Image &parent1, const Image &parent2)
	{
		int nImages = res->nImages;
	
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

	void CycleCrossover(Image &child1, Image &child2, const Image &parent1, const  Image &parent2)
	{
		int nImages = res->nImages;

		std::vector<int> p1(nImages);
		std::vector<int> p2(nImages);
		for(int i = 0; i < nImages; i++)
		{
			p1[i] = parent1.getImages()[i].first;
			p2[i] = parent2.getImages()[i].first;
		}
		
		const auto find = [&](const auto &self, const auto &p1, const auto &p2, int current, int startIndex, auto &fromP2) -> void
	    {
	        auto it = std::find(p2.begin(), p2.end(), current);
		    int currentIndex = it - p2.begin();
		    fromP2.push_back(currentIndex);
		    if(p1[currentIndex] == p1[startIndex]);
		   	else if(it != p2.end()) 
		    	self(self,p1,p2,p1[currentIndex],startIndex,fromP2); 
	    };
		
		std::vector<int> indexFromParent;
		indexFromParent.reserve(nImages);
		find(find,p1,p2,p1[0],0,indexFromParent);
		
		child1.setImages(parent2.getImages());
		for(auto p : indexFromParent)
			child1.setImage(p,parent1.getImages()[p]);
		
		indexFromParent.clear();
		indexFromParent.reserve(nImages);
		find(find,p2,p1,p2[0],0,indexFromParent);
		
		child2.setImages(parent1.getImages());
		for(auto p : indexFromParent)
			child2.setImage(p, parent2.getImages()[p]);
	}

	void SingleSwapMutation(Image &child)
	{
		int nImages = res->nImages;
		
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

	void InversionMutation(Image &child)
	{
		int nImages = res->nImages;
		
		int index1 = int(drand48()*nImages);
		int index2 = int(drand48()*nImages);
		while(index1 == index2)
			index2 = int(drand48()*nImages);

		int min = std::min(index1,index2);
		int max = std::max(index1,index2);
		std::reverse(child.getImages().begin() + min, child.getImages().begin() + max);
	}

	void ScrambleMutation(Image &child)
	{
		int nImages = res->nImages;
		
		int index1 = int(drand48()*nImages);
		int index2 = int(drand48()*nImages);
		while(index1 == index2)
			index2 = int(drand48()*nImages);

		int min = std::min(index1,index2);
		int max = std::max(index1,index2);
		child.shuffle(false,min,max);
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
	
	template <typename T>
	std::vector<size_t> SortIndexes(const std::vector<T> &v) 
	{
		std::vector<size_t> idx(v.size());
		std::iota(idx.begin(), idx.end(), 0);
		sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
		return idx;
	}
};