#include "VideoRepairer.hh"

int main(int ac, char** av)
{
    if (ac != 5 || std::string(av[1]) != "--video-path" 
        || av[2] == nullptr || std::string(av[3]) != "--output-name" ||
        av[4] == nullptr)
    {
        std::cerr << "Invalid arguments: ./videoRepairer --video-path video.mp4 --output-name output-video";
        return -1;  
    }

    const fs::path videoCorrupted(av[2]);
    if (!fs::exists(videoCorrupted))
    {
        std::cerr << "arg[2] must be a video path.";
        return -1;
    }
    const std::string outputName(av[4]);
    try
    {
        VideoRepairer   repairer(videoCorrupted, outputName);
    
        repairer.detectCorruptedFrames();
        auto & idFrames = repairer.sortFrames();

        repairer.createVideo(idFrames);

    }
    catch (std::exception& error)
    {
        std::cerr << error.what() << std::endl;
    }
    return 0;
}