#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Image
{
public:
	Image() {}
	Image(const Image &img) { images = img.images; }
	void add(int index, cv::Mat& img) { images.push_back(std::pair<int,cv::Mat>(index,img)); }
	void clear() { images.clear(); }
	
	cv::Mat& operator[] (int i) { return images[i].second; }
	int size() { return images.size(); }
	void shuffle() { std::random_shuffle(images.begin(), images.end());}
	
	std::vector<std::pair<int,cv::Mat> > getImages() { return images; }

	void printOrder() { for(auto i : images) printf("i=%d\n",i.first); 
}
private:
	std::vector<std::pair<int,cv::Mat> > images;
};