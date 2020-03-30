#pragma once
#ifndef VIDEOREPAIRER_HH_
# define VIDEOREPAIRER_HH_

# include <stdexcept>
# include <filesystem>
# include <opencv2/opencv.hpp>
# include <opencv2/tracking.hpp>

# include "AssignementManager.hh"

namespace   fs = std::filesystem;

using       pair_uint = std::pair<uint, uint>;
using       array_double = std::array<double, 3>;
using       const_it_multimap = std::multimap<double, pair_uint>::const_iterator;
using       frame_rec = std::pair<cv::Mat, cv::Rect2d>;
using       iterator_frames = std::unordered_map<uint, frame_rec>::iterator;

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
    const fs::path&                              _videoPath;
    std::unordered_map<uint, frame_rec>         _frames;

    std::vector<SampleSSIM>                     _samplesSSIM;
    double                                      _meanSSIMDist;
    double                                      _ecartType;
    std::vector<uint>                           _corruptedFrame;

    pair_uint                                   _pairIdsSim;

    std::unordered_map<uint,
        std::map<uint, double>>                 _IoUFrames;
    std::unordered_map<uint, uint>              _idFrameToIndex;
    std::unordered_map<uint, uint>              _indexToIdFrame;


    std::unordered_map<uint, uint>              _originalMapID;
    cv::Mat_<double>                            _originalCostMat;
    std::unique_ptr<Assignement>                _assginementFrames;

public:
    VideoRepairer(const fs::path &);
    ~VideoRepairer();
    
    void                                        detectCorruptedFrames();
    void                                        sortFrames();
    void                                        createVideo();

    const std::unordered_map<uint, frame_rec>&    getFrames()const;
            
private:
    void                                        _extractFrames();
    /*
    **  @brief: Checking the similarity between the two images.
    **  For each channel, a floating point number between 0 and 1 is computed (higher is better).
    **    
    **  @param1: hsv referencial frame in a Matrice Opencv data structure
    **  @param2: compared hsv frame in a Matrice Opencv data structure
    **  @return: SSIM in a Scalar OpenCV data structure
    */
    const cv::Scalar                            _getSSIM(const cv::Mat&,
                                                         const cv::Mat&) const;
    /*
    **@brief: Compute the SSIM between each frame that follows.
    **        The two frames are convert to HSV channel to use relevant data pixels.
    **        Call the getSSIM function to get their SSIM. A mean of those channels is computed 
    **        and stored in a SampleSSIM object with their ids.
    **        Each SampleSSIM object are stored in the _samplesSSIM vector.
    */
    void                                        _computeSSIM();
    /*
    ** @brief: Distinguish corrupted frames from the relevant ones by iterate over
    **         the associative z-score container. Those with a score inferior
    **         to the thresh (-0.5) are corrupt and therefore saved in the idCorrupted container.
    **         Than, to select the good one between the two id, we saved in the _corrupted frames 
    **         the one which have another occurence.
    **
    ** @return: ids pair of two good similar frames
    */
    const pair_uint                            _findCorruptedFrames();

    /*
    ** @brief: Compute the mean, the ecartype and all z-score of the distribution.
    **
    ** @return: associative container of z-score with their corresponding id.
    */
    const std::multimap<double,
                        pair_uint>              _computeDataGaussian();
    /*
    ** @brief: Distinguish the corrupted frame between the two frames that could not 
    **         be separated. By recomputing and comparing the SSIM between two frames 
    **         and a reference frame. The reference is the one with the highest SSIM 
    **         in the last calculation.
    **
    ** @param1: id reference frame
    ** @param2: container of corrupt ids detected 
    ** @param3: associative container of frames with their compared frames
    */
    void                                        _determineNewCorruptedFrame(const uint,
                                                                            std::multiset<uint>&,
                                                                            std::map<uint, std::vector<uint>> &);
    /*
    **  @brief: Remove corrupted frames in the _frames vector by their id saved in 
    **  the _corruptedFrame vector
    */
    void                                        _removeCorruptedFrames();

    void                                        _detectObj();
    void                                        _trackingObj();
    void                                        _computeIoU(iterator_frames &);
  
    void                                        _initMovementCost(cv::Mat_<double>&);
    void                                        _useHungarianAlgorithm(cv::Mat_<double>&);
    const std::vector<uint>                     _updateSetOfIdFrames();
    void                                        _updateCostMatrice(cv::Mat_<double>&);

    void                                        _showFrames();
    void                                        _showOrderederFrames( std::vector<std::list<uint>>&);
};

#endif // !VIDEOREPAIRER_HH_
