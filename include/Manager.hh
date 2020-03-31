#pragma once
#ifndef MANAGER_HH_
# define MANAGER_HH_

# include "ListAssignement.hh"

using iteratorList = uintList::iterator;
using iteratorVector = std::vector<uintList>::iterator;

class Manager
{
    std::vector<std::unique_ptr<ListAssignement>>       _assignLists;

    public:
        Manager();
        ~Manager();
        
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

#endif // !MANAGER_HH_
