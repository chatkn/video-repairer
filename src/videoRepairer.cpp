#include "VideoRepairer.hh"
#include "munkres.h"

VideoRepairer::VideoRepairer(const fs::path& videoPath, const std::string& outputName)
    : _videoPath(videoPath), _outputName(outputName)
{}

VideoRepairer::~VideoRepairer()
{
}

SampleSSIM::SampleSSIM(const pair_uint& ids,
                       const array_double& hsvSSIM,
                       const double meanHsvSSIM)
    : _ids(ids), _hsvSSIM(hsvSSIM), _meanHsvSSIM(meanHsvSSIM)
{}

SampleSSIM::~SampleSSIM(){}

void      VideoRepairer::_extractFrames()
{
    auto video = cv::VideoCapture(_videoPath.string());
    if (!video.isOpened())
        throw std::runtime_error("Error by opening the video " + _videoPath.filename().string());

    uint frame_width = static_cast<uint>(video.get(cv::CAP_PROP_FRAME_WIDTH));
    uint frame_height = static_cast<uint>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
    _frameSize = cv::Size(frame_width, frame_height);
    _fourcc = static_cast<int>(video.get(cv::CAP_PROP_FOURCC));
    _fps = video.get(cv::CAP_PROP_FPS);

    cv::Mat frame;
    uint    id = 0;
    while (video.read(frame))
    {
        frame_rec frameRec = std::make_pair(frame.clone(), cv::Rect2d());
        _frames.insert(std::make_pair(id, frameRec));
        ++id;
    }

    video.release();
}

const cv::Scalar    VideoRepairer::_getSSIM(const cv::Mat& i1,
                                         const cv::Mat& i2) const
{
    int d = CV_32F;

    cv::Mat I1, I2;
    i1.convertTo(I1, d);           // cannot calculate on one byte large values
    i2.convertTo(I2, d);

    cv::Mat I2_2 = I2.mul(I2);        // I2^2
    cv::Mat I1_2 = I1.mul(I1);        // I1^2
    cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

    cv::Mat mu1, mu2;   //
    cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

    cv::Mat mu1_2 = mu1.mul(mu1);
    cv::Mat mu2_2 = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_2, sigma2_2, sigma12;

    cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;

    cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;

    cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;

    cv::Mat t1, t2, t3;
    const double C1 = 6.5025, C2 = 58.5225;

    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

    cv::Mat ssim_map;
    cv::divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

    return  mean(ssim_map); // mssim = average of ssim map
}

void    VideoRepairer::_computeSSIM()
{
    cv::Mat hsvFrameFst, hsvFrameSnd;
    auto itFirst = _frames.begin();
    auto itSnd = itFirst;

    while (itFirst != _frames.end() && std::next(itFirst) != _frames.end())
    {
        itSnd = std::next(itFirst, 1);
        const auto & frameFirst = (*itFirst);
        const auto & frameSnd = (*itSnd);

        cv::cvtColor(frameFirst.second.first.clone(), hsvFrameFst, cv::COLOR_BGR2HSV);
        cv::cvtColor(frameSnd.second.first.clone(), hsvFrameSnd, cv::COLOR_BGR2HSV);

        const auto& ssimHSV = _getSSIM(hsvFrameFst.clone(), hsvFrameSnd.clone());
        const array_double hsvSSIM{ ssimHSV[0]*100, ssimHSV[1] * 100, ssimHSV[2] * 100 };
        const double meanHsvSSIM = round((hsvSSIM[0] + hsvSSIM[1] + hsvSSIM[2]) / 3);

        const pair_uint ids{ frameFirst.first, frameSnd.first };
        SampleSSIM sample(ids, hsvSSIM, meanHsvSSIM);

        _samplesSSIM.push_back(sample);

        itFirst = itSnd;
    }
}

void    VideoRepairer::_removeCorruptedFrames()
{
    for (auto& id : _corruptedFrame)
    {
        auto& it = _frames.find(id);
        it = _frames.erase(it);
    }

}

const std::multimap<double, 
                    pair_uint>      VideoRepairer::_computeDataGaussian()
{
    double  meanSSIMDist = 0;
    double  ecartType = 0;

    /*      MEAN ON DISTRIB            */
    for (const auto & sample : this->_samplesSSIM)
    {
        const double& meanHsv = sample._meanHsvSSIM;
        meanSSIMDist += meanHsv;
    }
    meanSSIMDist = round(meanSSIMDist / this->_samplesSSIM.size());

    /*      ECART TYPE ON DISTRIB      */
    double sigma = 0;
    for (const auto & sample: this->_samplesSSIM)
    {
         const auto dist = (sample._meanHsvSSIM - meanSSIMDist);
         sigma += (dist * dist);
    }
    ecartType = round(std::sqrt(sigma) / this->_samplesSSIM.size());

    /*      Z-SCORE DISTRIB            */
    std::multimap<double, pair_uint>   zScoreOrdered;
    for (auto & sample : this->_samplesSSIM)
    {
        sample._zScore = (sample._meanHsvSSIM - meanSSIMDist) / ecartType;
        zScoreOrdered.insert(std::make_pair(sample._zScore, sample._ids));
    }

    return zScoreOrdered;
}

void    VideoRepairer::_findCorruptedFrames()
{
    const auto & zScoreSamples = this->_computeDataGaussian();

    std::multiset<uint>                 idsCorrupted;
    std::map<uint, std::vector<uint>>   comparedFrames;

    /*  Get outlier frames */
    for (const auto& distribution : zScoreSamples)
    {
        const auto fstIdDistrib = distribution.second.first;
        const auto sndIdDistrib = distribution.second.second;

        if (distribution.first < -0.5) // thresh = -0.5 by analyzing the zScores
        {
            comparedFrames[fstIdDistrib].push_back(sndIdDistrib);
            comparedFrames[sndIdDistrib].push_back(fstIdDistrib);
            idsCorrupted.insert(fstIdDistrib);
            idsCorrupted.insert(sndIdDistrib);
        }
    }
    /*  detect the id outlier by store the ones with occurence */
    for (auto & it = idsCorrupted.begin(); it != idsCorrupted.end(); ++it)
    {
        it = std::adjacent_find(it, idsCorrupted.end());
        if (it == idsCorrupted.end())
            break;
        this->_corruptedFrame.push_back(*it);
    }
    const auto & idRef = std::prev(zScoreSamples.end(), 1);
    this->_determineNewCorruptedFrame(idRef->second.first, idsCorrupted, comparedFrames);
    const auto & pairIdSim = std::prev(zScoreSamples.end(), 4);

    _pairIdsSim = pairIdSim->second;
}

void    VideoRepairer::_determineNewCorruptedFrame(const uint idRef,
                                                   std::multiset<uint>& idsCorrupted, 
                                                   std::map<uint, std::vector<uint>> &comparedFrames)
{
    for (const auto & id : this->_corruptedFrame)
    {
        idsCorrupted.erase(id);
        const auto & vectId = comparedFrames[id];
        for (const auto & associativeId : vectId)
            idsCorrupted.erase(associativeId);
    }
    
    if (!idsCorrupted.empty())
    {
        std::map<double, uint>  newSSIMIds;

        for (const auto & id : idsCorrupted)
        {
            const auto ssimHSV = _getSSIM(_frames[idRef].first.clone(), _frames[id].first.clone());
            const double meanSSIMHsv = (ssimHSV[0] + ssimHSV[1] + ssimHSV[2]) / 3;

            newSSIMIds.insert(std::make_pair(meanSSIMHsv * 100, id));
        }
        const auto& itCorrupted = newSSIMIds.begin();
        this->_corruptedFrame.push_back(itCorrupted->second);
    }
}

void    VideoRepairer::_detectObj()
{
    const uint idFirst = _pairIdsSim.first;
    cv::Mat frameOne = _frames[idFirst].first.clone();
    cv::Mat frameTwo = _frames[_pairIdsSim.second].first.clone();

    cv::GaussianBlur(frameOne.clone(), frameOne, cv::Size(3, 3), 0);
    cv::GaussianBlur(frameTwo.clone(), frameTwo, cv::Size(3, 3), 0);

    cv::cvtColor(frameOne.clone(), frameOne, cv::COLOR_BGR2HSV);
    cv::cvtColor(frameTwo.clone(), frameTwo, cv::COLOR_BGR2HSV);

    cv::Mat diff;
    cv::absdiff(frameTwo, frameOne, diff);

    std::vector<cv::Mat> hsvSplit, contours;
    cv::split(diff, hsvSplit);

    cv::Mat satChannel = hsvSplit[1];
    cv::Mat shapeElement = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(3, 3));

    cv::morphologyEx(satChannel.clone(), satChannel, cv::MORPH_CLOSE, shapeElement, cv::Point(-1, -1), 10);
    cv::threshold(satChannel.clone(), satChannel, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
    cv::findContours(satChannel, contours, cv::RetrievalModes::RETR_EXTERNAL, 
                    cv::ContourApproximationModes::CHAIN_APPROX_SIMPLE);

    std::map<double, const cv::Rect2d> areaContours;
    for (auto contour : contours)
    {
        const cv::Rect2d rec = cv::boundingRect(contour);
        areaContours.insert(std::make_pair(rec.area(), rec));
    }
 
    const auto higherContour = areaContours.rbegin();
    _frames[idFirst].second = higherContour->second;
    _trackingObj();

}

void    VideoRepairer::_trackingObj()
{
    const uint idInit = _pairIdsSim.first;
    cv::Ptr<cv::Tracker> tracker = cv::TrackerCSRT::create();

    cv::Mat frameInit = _frames[idInit].first;
    auto & roi = _frames[idInit].second;
    cv::Mat frameRec = frameInit.clone();
    cv::rectangle(frameRec, roi, cv::Scalar(255, 0, 0), 2);
    tracker->init(frameInit, roi);
    std::vector<uint>   _noTrackedFrame;
    for (auto& dataframe : _frames)
    {
        if (dataframe.first == idInit)
            continue;
        cv::Rect2d& detectRoi = dataframe.second.second;
        cv::Mat frame = dataframe.second.first;
        tracker->update(frame, detectRoi);
        if (detectRoi.area() == 0)
            _noTrackedFrame.push_back(dataframe.first);
    }

    for (auto &id : _noTrackedFrame)
    {
        auto &it = _frames.find(id);
        _frames.erase(it);
    }
}

void    VideoRepairer::_computeIoU(iterator_frames& itFirst)
{
    const auto& boxFirst = itFirst->second.second;
    cv::Mat     firstFrame = itFirst->second.first.clone();
    cv::Mat     sndFrame, diff;
    std::vector<uint>   sameFrameToRemove;
    std::map<uint, double>  IuOWithdId;
    for (auto & snd : _frames)
    {
        if (itFirst->first == snd.first)
            continue;

        const auto& boxSnd = snd.second.second;
        const auto interAreaBox = (boxFirst & boxSnd).area();
        const auto unionAreaBox = (boxFirst.area() + boxSnd.area()) - interAreaBox;
        const auto IoU = interAreaBox / unionAreaBox;

        IuOWithdId.insert(std::make_pair(snd.first, IoU));
    }

    _IoUFrames.insert(std::make_pair(itFirst->first, IuOWithdId));
    if (std::next(itFirst, 1) != _frames.end())
    {
        itFirst = std::next(itFirst, 1);
        _computeIoU(itFirst);
    }
}

void    VideoRepairer::_initMovementCost(cv::Mat_<double>& hugarianMatrice)
{
    /*      INIT        */
    uint    id = 0;
    for (const auto& _IoUFrame : _IoUFrames)
    {
        _idFrameToIndex.insert(std::make_pair(_IoUFrame.first, id));
        _indexToIdFrame.insert(std::make_pair(id++, _IoUFrame.first));
    }

    _originalMapID = _idFrameToIndex;
    for (auto& _IoUFrame : _IoUFrames)
    {
        uint firstId = _IoUFrame.first;
        const auto & idIoU = _IoUFrame.second;
        const uint row = _idFrameToIndex[firstId];
        for (const auto& pair : idIoU)
        {
            uint sndId = pair.first;
            const double IoU = pair.second;
            const uint col = _idFrameToIndex[sndId];
            double value = (1 - IoU) * 10;
            double roundDecimal = (ceil((value * 100)) / 100);
            hugarianMatrice[row][col] = roundDecimal;
        }
    }
    _originalCostMat = hugarianMatrice.clone();
}

void    VideoRepairer::_useHungarianAlgorithm(cv::Mat_<double>& hugarianMatrice)
{
    Munkres m;
    m.diag(false);
    m.solve(hugarianMatrice);
}

const std::vector<uint>     VideoRepairer::_updateSetOfIdFrames()
{
    const auto& idsFrame = this->_assginementFrames->getExtremityIdClose();

    uint index = 0;
    _indexToIdFrame.clear();
    _idFrameToIndex.clear();

    for (auto& idFrame : idsFrame)
    {
        _idFrameToIndex.insert(std::make_pair(idFrame, index));
        _indexToIdFrame.insert(std::make_pair(index++, idFrame));
    }

    return idsFrame;
}

void    VideoRepairer::_updateCostMatrice(cv::Mat_<double>& hungarianMat)
{
    const auto& idsFrame = _updateSetOfIdFrames();
    const uint  size = static_cast<uint>(idsFrame.size());
    cv::resize(hungarianMat, hungarianMat, cv::Size(size, size));
    hungarianMat.setTo(10);

    for (auto& itRow = idsFrame.begin(); itRow != idsFrame.end(); ++itRow)
    {
        auto& vecIoU = _IoUFrames[*itRow];
        for (auto& idCol : idsFrame)
        {
            if (idCol == *itRow || (std::next(itRow, 1) != idsFrame.end()
                && idCol == *std::next(itRow, 1)))
                continue;

            uint originalIdRow = _originalMapID[*itRow];
            uint originalIdCol = _originalMapID[idCol];
            const double cost = _originalCostMat[originalIdRow][originalIdCol];

            uint indexRow = _idFrameToIndex[*itRow];
            uint indexcol = _idFrameToIndex[idCol];
            hungarianMat[indexRow][indexcol] = cost;
        }
    }
}

void    VideoRepairer::detectCorruptedFrames()
{
    this->_extractFrames();
    std::cout << "[Video Repairer] frames extracted from the video" << std::endl;

    std::cout << "[Video Repairer] Starting to compute SSIM between frames..." << std::endl;
    this->_computeSSIM();
    std::cout << "[Video Repairer] ...a SSIM is computed between each image that follows" << std::endl;

    this->_findCorruptedFrames();
    this->_removeCorruptedFrames();
    std::cout << "[Video Repairer] Corrupted frames were found and removed" << std::endl;
}

const std::list<uint>   VideoRepairer::sortFrames()
{
    this->_detectObj();
    std::cout << "[Video Repairer] Object detected and tracked from frames" << std::endl;

    this->_computeIoU(_frames.begin());
    std::cout << "[Video Repairer] Computed IoU of the detected object between each frames" << std::endl;

    const uint size = static_cast<uint>(_IoUFrames.size());
    cv::Mat_<double> hugarianMatrice(size, size, 10); // 10 for max default cost

   /*
    *   INIT
    */
    this->_initMovementCost(hugarianMatrice);
    std::cout << "[Video Repairer] Applied a movement cost for each IoU values" << std::endl;
    this->_useHungarianAlgorithm(hugarianMatrice);
    std::cout << "[Video Repairer] Computed the hungarian matrice from the cost matrice" << std::endl;

    this->_assginementFrames = std::make_unique<Manager>();
    std::cout << "[Video Repairer] The manager of frames assignement is launched" << std::endl;
    this->_assginementFrames->launchMatching(hugarianMatrice, this->_indexToIdFrame);

   /*
    *   UPDATE
    */
    for (auto& closeList = this->_assginementFrames->getCloseList(0);
        closeList.size() != 1;)
    {
        this->_updateCostMatrice(hugarianMatrice);
        this->_useHungarianAlgorithm(hugarianMatrice);
        this->_assginementFrames->launchMatching(hugarianMatrice, this->_indexToIdFrame);
    }

    std::cout << "[Video Repairer] Finish to compute the last round of frames assignement" << std::endl;
    return this->_assginementFrames->getCloseList(0)[0];
}

void    VideoRepairer::createVideo(const uintList& idList)
{
    auto newpath = _videoPath;
    newpath.replace_filename(_outputName + ".mp4");
    uint id = 1;
    while (fs::exists(newpath))
    {
        newpath.replace_filename(_outputName + std::to_string(id) + ".mp4");
        ++id;
    }
    cv::VideoWriter outputVideo(newpath.string(), _fourcc, _fps, _frameSize, true);

    if (!outputVideo.isOpened())
    {
        std::cout << "Could not open the output video for write: " << std::endl;
        return;
    }

     for (const auto & id : idList)
    {
        cv::Mat frame = _frames[id].first;
        outputVideo.write(frame);
    }
    std::cout << "[Video Repairer] the video " << newpath.filename().string() << " is created" << std::endl;
    std::cout << "[Video Repairer] Access path: " << newpath.parent_path().string() << std::endl;
}