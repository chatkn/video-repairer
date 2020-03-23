#include "VideoRepairer.hh"

VideoRepairer::VideoRepairer(const fs::path& videoPath)
    : _videoPath(videoPath), _meanSSIMDist(0)
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

void    VideoRepairer::_extractFrames()
{
    auto video = cv::VideoCapture(_videoPath.string());
    if (!video.isOpened())
        throw std::runtime_error("Error by opening the video " + _videoPath.filename().string());

    cv::Mat frame;
    uint    id = 0;
    while (video.read(frame))
    {
        _frames.insert(std::make_pair(id, frame.clone()));
        ++id;
    }

    video.release();
}

const cv::Scalar VideoRepairer::_getSSIM(const cv::Mat& i1,
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

/*
    compute ssim between each frame that follows because the reference frame is the
    precedent frame.
*/
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

        cv::cvtColor(frameFirst.second.clone(), hsvFrameFst, cv::COLOR_BGR2HSV);
        cv::cvtColor(frameSnd.second.clone(), hsvFrameSnd, cv::COLOR_BGR2HSV);

        const auto& ssimHSV = _getSSIM(hsvFrameFst.clone(), hsvFrameSnd.clone());
        const array_double hsvSSIM{ ssimHSV[0]*100, ssimHSV[1] * 100, ssimHSV[2] * 100 };
        const double meanHsvSSIM = round((hsvSSIM[0] + hsvSSIM[1] + hsvSSIM[2]) / 3);

        const pair_uint ids{ frameFirst.first, frameSnd.first };
        SampleSSIM sample(ids, hsvSSIM, meanHsvSSIM);

        _samplesSSIM.push_back(sample);

        itFirst = itSnd;
    }
}

void    VideoRepairer::_showFrames()
{
    cv::namedWindow("frame", cv::WINDOW_NORMAL);
    for (auto frame : _frames)
    {
        cv::imshow("frame", frame.second.clone());
        cv::waitKey(0);
    }
    cv::destroyWindow("frame");    
}

void        VideoRepairer::_removeCorruptedFrames()
{
    for (auto id : _corruptedFrame)
    {
        auto it = _frames.find(id);
        it = _frames.erase(it);
    }

}

const std::multimap<double, pair_uint>        VideoRepairer::_computeDataGaussian()
{
    /*      MEAN ON DISTRIB            */
    for (const auto & sample : this->_samplesSSIM)
    {
        const double& meanHsv = sample._meanHsvSSIM;
        this->_meanSSIMDist += meanHsv;
    }
    this->_meanSSIMDist = round(this->_meanSSIMDist / this->_samplesSSIM.size());

    /*      ECART TYPE ON DISTRIB      */
    double sigma = 0;
    for (const auto & sample: this->_samplesSSIM)
    {
         const auto dist = (sample._meanHsvSSIM - this->_meanSSIMDist);
         sigma += (dist * dist);
    }
    this->_ecartType = round(std::sqrt(sigma) / this->_samplesSSIM.size());

    /*      Z-SCORE DISTRIB            */
    std::multimap<double, pair_uint>   zScoreOrdered;
    for (auto & sample : this->_samplesSSIM)
    {
        sample._zScore = (sample._meanHsvSSIM - this->_meanSSIMDist) / this->_ecartType;
        zScoreOrdered.insert(std::make_pair(sample._zScore, sample._ids));
    }

    return zScoreOrdered;
}

void        VideoRepairer::_findCorruptedFrames()
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
    /*  detect the id outlier by store ones with occurence */
    for (auto & it = idsCorrupted.begin(); it != idsCorrupted.end(); ++it)
    {
        it = std::adjacent_find(it, idsCorrupted.end());
        if (it == idsCorrupted.end())
            break;
        this->_corruptedFrame.push_back(*it);
    }
    this->_determineNewCorruptedFrame(std::prev(zScoreSamples.end(), 1), idsCorrupted, comparedFrames);

}

/*
        insert the frame that look like corrupted by recompute their ssim
        with a reference frame that is certainly not corrupted.

*/
void    VideoRepairer::_determineNewCorruptedFrame(const const_it_multimap sampleRef,
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
        const auto& idRef = (*sampleRef).second;
        std::map<double, uint>  newSSIMIds;

        for (const auto & id : idsCorrupted)
        {
            const auto ssimHSV = _getSSIM(_frames[idRef.first].clone(), _frames[id].clone());
            const double meanSSIMHsv = (ssimHSV[0] + ssimHSV[1] + ssimHSV[2]) / 3;

            newSSIMIds.insert(std::make_pair(meanSSIMHsv * 100, id));
        }
        const auto& itCorrupted = newSSIMIds.begin();
        this->_corruptedFrame.push_back(itCorrupted->second);
    }
}

void    VideoRepairer::startRepair()
{
    this->_extractFrames();
    std::cout << "[Video Repairer] frames extracted from the video" << std::endl;

    std::cout << "[Video Repairer] Starting to compute SSIM between frames..." << std::endl;
    this->_computeSSIM();
    std::cout << "[Video Repairer] ...a SSIM is computed between each image that follows." << std::endl;

    this->_findCorruptedFrames();
    std::cout << "[Video Repairer] Corrupted frames was found" << std::endl;

    this->_removeCorruptedFrames();
    std::cout << "[Video Repairer] Corrupted frames was removed" << std::endl;

    this->_showFrames();
}

const std::unordered_map<uint, cv::Mat>&     VideoRepairer::getFrames()const
{
    return _frames;
}
