#pragma once
#ifndef VIDEOREPAIRER_HH_
# define VIDEOREPAIRER_HH_

# include <iostream>
# include <stdexcept>
# include <filesystem>
# include <opencv2/opencv.hpp>


namespace   fs = std::filesystem;

using       pair_uint = std::pair<uint, uint>;
using       array_double = std::array<double, 3>;
using       const_it_multimap = std::multimap<double, pair_uint>::const_iterator;

struct                          SampleSSIM
{
    const pair_uint            _ids;
    const array_double         _hsvSSIM;
    const double               _meanHsvSSIM;
    double                     _zScore;

    SampleSSIM(const pair_uint&, const array_double&,
        const double meanHsvSSIM);
    ~SampleSSIM();
};


class                                           VideoRepairer
{
    const fs::path&                             _videoPath;
    std::unordered_map<uint, cv::Mat>           _frames;
    std::vector<SampleSSIM>                     _samplesSSIM;
    double                                      _meanSSIMDist;
    double                                      _ecartType;
    std::vector<uint>                           _corruptedFrame;

public:
    VideoRepairer(const fs::path &);
    ~VideoRepairer();

    void                                        startRepair();
    const std::unordered_map<uint, cv::Mat>&    getFrames()const;

private:
    void                                        _extractFrames();
    const cv::Scalar                            _getSSIM(const cv::Mat&,
                                                         const cv::Mat&) const;
    void                                        _computeSSIM();
    void                                        _findCorruptedFrames();
    const std::multimap<double,
                        pair_uint>              _computeDataGaussian();
    void                                        _determineNewCorruptedFrame(const const_it_multimap,
                                                                            std::multiset<uint>&,
                                                                            std::map<uint, std::vector<uint>> &);
    void                                        _removeCorruptedFrames();
    void                                        _showFrames();
 
};

#endif // !VIDEOREPAIRER_HH_
