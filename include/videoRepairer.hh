#pragma once
#ifndef VIDEOREPAIRER_HH_
# define VIDEOREPAIRER_HH_

# include <stdexcept>
# include <filesystem>
# include <opencv2/opencv.hpp>
# include <opencv2/tracking.hpp>

# include "Manager.hh"

namespace   fs = std::filesystem;

using       pair_uint = std::pair<uint, uint>;
using       array_double = std::array<double, 3>;
using       const_it_multimap = std::multimap<double, pair_uint>::const_iterator;
using       frame_rec = std::pair<cv::Mat, cv::Rect2d>;
using       iterator_frames = std::unordered_map<uint, frame_rec>::iterator;

struct                      SampleSSIM
{
    const pair_uint         _ids;
    const array_double      _hsvSSIM;
    const double            _meanHsvSSIM;
    double                  _zScore;

    SampleSSIM(const pair_uint&, 
               const array_double&,
               const double);
    ~SampleSSIM();
};


class                               VideoRepairer
{
    const fs::path&                 _videoPath;
    const std::string&              _outputName;
    cv::Size                        _frameSize;
    int                             _fourcc;
    double                          _fps;

    std::unordered_map<uint, 
                       frame_rec>   _frames;
    std::vector<SampleSSIM>         _samplesSSIM;
    std::vector<uint>               _corruptedFrame;
    pair_uint                       _pairIdsSim;

    std::unordered_map<uint,
        std::map<uint, double>>     _IoUFrames;
    std::unordered_map<uint, uint>  _idFrameToIndex;
    std::unordered_map<uint, uint>  _indexToIdFrame;


    std::unordered_map<uint, uint>  _originalMapID;
    cv::Mat_<double>                _originalCostMat;
    std::unique_ptr<Manager>        _assginementFrames;

public:
    VideoRepairer(const fs::path &, 
                 const std::string&s);
    ~VideoRepairer();
    
    void                            detectCorruptedFrames();
    const std::list<uint>           sortFrames();
    void                            createVideo(const uintList&);
            
private:
    void                            _extractFrames();
    /*
    **  @brief: Checking the similarity between the two images.
    **  For each channel, a floating point number between 0 and 1 is computed (higher is better).
    **    
    **  @param1: hsv referencial frame in a Matrice Opencv data structure
    **  @param2: compared hsv frame in a Matrice Opencv data structure
    **  @return: SSIM in a Scalar OpenCV data structure
    */
    const cv::Scalar                _getSSIM(const cv::Mat&,
                                             const cv::Mat&) const;
    /*
    **@brief: Compute the SSIM between each frame that follows.
    **        The two frames are convert to HSV channel to use relevant data pixels.
    **        Call the getSSIM function to get their SSIM. A mean of those channels is computed 
    **        and stored in a SampleSSIM object with their ids.
    **        Each SampleSSIM object are stored in the _samplesSSIM vector.
    */
    void                            _computeSSIM();
    /*
    ** @brief: Distinguish corrupted frames from the relevant ones by iterate over
    **         the associative z-score container. Those with a score inferior
    **         to the thresh (-0.5) are corrupt and therefore saved in the idCorrupted container.
    **         Than, to select the good one between the two id, we saved in the _corrupted frames 
    **         the one which have another occurence.
    **         Set ids pair of two good similar frames. 
    */
    void                            _findCorruptedFrames();

    /*
    ** @brief: Compute the mean, the ecartype and all z-score of the distribution.
    **
    ** @return: associative container of z-score with their corresponding id.
    */
    const std::multimap<double,
                        pair_uint>  _computeDataGaussian();
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
    void                            _determineNewCorruptedFrame(const uint,
                                                                std::multiset<uint>&,
                                                                std::map<uint, std::vector<uint>>&);
    /*
    **  @brief: Remove corrupted frames in the _frames vector by their id saved in 
    **  the _corruptedFrame vector
    */
    void                            _removeCorruptedFrames();

    /*
    **  @brief: Compare the two most similar frames (_pairIdsSim) to detect the moving object.
    **          Convert frames to HSV channel and apply morphology transformations
    **          to highlight the moving contours. The contour with the highest area
    **          is the one of the moving object searched.
    */
    void                            _detectObj();

    /*
    **  @brief: Use the TrackerCSRT of opencv to determine in each frame the
    **          roi of the moving object by providing first the initial roi 
    **          and its corresponding frame.
    */
    void                            _trackingObj();

    /*
    **  @brief: Compute IoU (intersection over union) between each frames
    **  @param1: iterator to the frame.
    */
    void                            _computeIoU(iterator_frames &);
  
    /*
    **  @brief: Compute the movement cost between each frame and store the inital matrice in
    **          _originalCost Mat.
    **  @param1: hungarian matrice to fill.
    */
    void                            _initMovementCost(cv::Mat_<double>&);
    
    /*
    **  @brief: Use the Hungarian algorithm implemented by John Weaver.
    **          Update the hungarian matrice by determine matching frames and unmatching frames.
    **          to minimize the cost of the moving object
    **  @param1: hungarian matrice to fill.
    */
    void                            _useHungarianAlgorithm(cv::Mat_<double>&);

    /*
    **  @brief: Check if initials matching fit by compute their difference of pixels
    **  @param1: initial matching ids.
    **  @param2: matching ids to erase.
    **  @param3: vector of matching ids to swap.
    */
    void                            _checkMatching(const std::vector<std::list<uint>>&,
                                                   std::map<uint, std::vector<uint>>&,
                                                   std::vector<uint>&);

    /*
    **  @brief: Compute nb of difference pixels bewteen matching ids and determine 
    **          if its relevant to separate them or swap their order.
    **  @param1: frame A with roi ojbject  
    **  @param2: frame B with roi ojbject
    **  @return: boolean 1: ids no longer match, boolean2: ids need to swap
    */
    const std::array<bool, 2>       _is_initialMatchingValid(cv::Mat,
                                                             cv::Mat);

    /*
    **  @brief: Compute nb of difference pixels
    **  @param2: matrice of difference between the 2 frames
    **  @return: percentage of white pixels
    */
    const float                     _computePixelsDifference(const cv::Mat);

    /*
    **  @brief: FilL the idsToErase container by id that no longer match
    **  @param1: id of frame A
    **  @param2: id of frame B or -1
    **  @param3: vector to fill
    **  @param4: index line where ids are in their close list
    */
    void                            _separateIds(uint, 
                                                 int,
                                                 std::map<uint, std::vector<uint>>&,
                                                 uint);
    /*
    **  @brief: Update hungarian matrice index and ids frame vector to match
    **  @return: vector of ids frame
    */
    const std::vector<uint>         _updateData();

    /*
    **  @brief: Update the hungarian matrice cost
    **  @param1: hungarian matrice
    */
    void                            _updateCostMatrice(cv::Mat_<double>&);
};

#endif // !VIDEOREPAIRER_HH_
