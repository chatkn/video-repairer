#pragma once
#ifndef LISTASSIGNEMENT_HH_
# define LISTASSIGNEMENT_HH_

# include <iostream>
# include <algorithm>
# include <list>
# include <opencv2/core/mat.hpp>
# include <unordered_map>

using uintList = std::list<uint>;
enum Direction { NONE, BACK, FRONT };


class ListAssignement
{
    const cv::Mat_<double>              _hungarianMatrice;
    const std::unordered_map<uint,
                            uint>       _indexToIdFrame;
    std::vector<uintList>               _close;
    std::vector<uintList>               _open;
    std::vector<uint>                   _extremityIdClose;

public:

    ListAssignement(cv::Mat_<double>&, 
                    const std::unordered_map<uint, uint>&);

    ListAssignement(const std::vector<uintList>&);
    ~ListAssignement();

    void                                   sort();
    void                                   updateIndexToIdFrame();
    std::pair<Direction, uintList>         findId(const uint);
    const std::vector<uintList>&           getCloseList();
    const std::vector<uint>                getExtremityIdClose();

private:
    const std::array<bool, 2>               _areTheyAlreadyAdded(const uint, 
                                                                 const uint);
};

#endif // !LISTASSIGNEMENT_HH_
