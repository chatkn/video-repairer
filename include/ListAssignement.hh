#pragma once
#ifndef LISTASSIGNEMENT_HH_
# define LISTASSIGNEMENT_HH_

# include <iostream>
# include <algorithm>
# include <list>
# include <opencv2/core/mat.hpp>
# include <unordered_map>
# include <map>


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
    
    /*
    ** @brief: Sort frames by finding match in the hungariab matrice.
    ** ids with a cost of 0 match and are stored in an open list.
    ** Check if ids are already added to close the open list.
    */
    void                                   sort();
    void                                   updateIndexToIdFrame();

    /*
    ** @brief: Find id and its direction in the close list with its matching ids.
    ** @param1: id to find
    ** @return: pair of id direction and its matching ids list
    */
    std::pair<Direction, uintList>         findId(const uint);

    const std::vector<uintList>&           getCloseList();

    /*
    ** @brief: From the close matching list, get the front and back id to find a new match.
    ** @return: vector of ids extremities of the last list of matched frame created.
    */
    const std::vector<uint>                getExtremityIdClose();
    void                                   setCloseList(const std::map<uint, std::vector<uint>>& idToErase,
                                                        const std::vector<uint>& swap);

private:

    /*
    ** @brief: Check if ids are already added in the open list
    **         to move them in the close list or add them next to
    **         their matching id.
    ** @param1: row id
    ** @param2: col id
    */
    const std::array<bool, 2>               _areTheyAlreadyAdded(const uint,
                                                                 const uint);
};

#endif // !LISTASSIGNEMENT_HH_
