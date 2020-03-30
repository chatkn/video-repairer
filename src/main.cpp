#include "VideoRepairer.hh"

int main(int ac, char** av)
{
    if (ac != 3 || std::string(av[1]) != "--video-path" 
        || av[2] == nullptr)
    {
        std::cerr << "Invalid arguments: ./videoRepairer --video-path video.mp4";
        return -1;  
    }

    const fs::path videoCorrupted(av[2]);
    if (!fs::exists(videoCorrupted))
    {
        std::cerr << "arg[2] must be a video path.";
        return -1;
    }

    try
    {
        VideoRepairer   repairer(videoCorrupted);
    
        repairer.detectCorruptedFrames();
        repairer.sortFrames();
        repairer.createVideo();

    }
    catch (std::exception& error)
    {
        std::cerr << error.what() << std::endl;
    }
    return 0;
}