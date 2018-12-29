#pragma once

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

class Image
{
public:
	Image() {}
	void Add(int index, cv::Mat img) { images.push_back(std::pair<int,cv::Mat>(index,img)); }
	void Clear() { images.clear(); }
private:
	std::vector<std::pair<int,cv::Mat> > images;
};