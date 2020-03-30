#pragma once

# include <iostream>
# include <algorithm>
# include <list>
# include <opencv2/core/mat.hpp>
# include <unordered_map>


    using uintList = std::list<uint>;
    using iteratorList = uintList::iterator;
    using iteratorVector = std::vector<uintList>::iterator;

    enum Direction { NONE, BACK, FRONT };

    class List
    {
        std::vector<uintList>               _close;
        std::vector<uintList>               _open;
        cv::Mat_<double>                    _hungarianMatrice;
        std::unordered_map<uint, uint>      _indexToIdFrame;
        std::vector<uint>                   _extremityIdClose;

    public:

        List(cv::Mat_<double>&, const std::unordered_map<uint, uint>&);
        List(const std::vector<uintList>&);
        ~List();

        void                                   sort();
        void                                   updateIndexToIdFrame();
        std::pair<Direction, uintList>         findId(const uint);
        const std::vector<uintList>&           getCloseList() const;
        const std::vector<uint>                getExtremityIdClose();

    private:
        const std::array<bool, 2>               _areTheyAlreadyAdded(const uint, 
                                                                     const uint);
    };

    class Assignement
    {
        std::vector<std::unique_ptr<List>>       _assignLists;

    public:
        Assignement();
        ~Assignement();

        void                                    launchMatching(cv::Mat_<double>&,
                                                              const std::unordered_map<uint, uint>&);
        const std::vector<uintList>&            getCloseList(const uint id) const;
        const std::vector<uint>                 getExtremityIdClose();

    private:
        void                                    _updateLists(std::vector<uintList>&);
        void                                    _mergeIDs(uint, iteratorList&, iteratorList&, 
                                                          iteratorVector&, std::pair<Direction, uintList>&);

        const std::pair<iteratorVector, iteratorList>          _foundElemNewList(const uint id, std::vector<uintList>&);
    };