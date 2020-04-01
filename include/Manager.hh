#pragma once
#ifndef MANAGER_HH_
# define MANAGER_HH_

# include "ListAssignement.hh"

using iteratorList = uintList::iterator;
using iteratorVector = std::vector<uintList>::iterator;
using listAssignementVector = std::vector<std::unique_ptr<ListAssignement>>;

class                                   Manager
{
        listAssignementVector           _assignLists;

    public:
        Manager();
        ~Manager();
        
        /*
        **  @brief: Create a new ListAssignement pointer and start sorting frames.
        **          When there is two list, the manager merge them to a new and erase
        **          the others of its storage.
        **  @param1: Hungarian matrice
        **  @param2: map of hungarian matrice index and frames index
        */
        void                            launchMatching(cv::Mat_<double>&,
                                                       const std::unordered_map<uint, uint>&);
        const std::vector<uintList>&    getCloseList(const uint id) const;

        /*
        ** @brief: Get the last listAssignement created and from its close list,
        **         create a vector of its front and back id to find a new match.
        ** @return: vector of ids extremities of the last list of matched frame created.
        */
        const std::vector<uint>         getExtremityIdClose();

        /*
        ** @brief: update the matched frames list of the given list index.
        ** @param1: id of the listAssignement to update
        ** @param2: vector of ids to unmatch
        ** @param3: vector of ids to swap
        */
        void                            setCloseList(const uint,
                                                     const std::map<uint, std::vector<uint>>&,
                                                     const std::vector<uint>&);

    private:

        /*
        ** @brief: Merge the two match list in a new one.
        **         By looping over each matching list of the second listAssignement and for each
        **         id, integrate its existing match (found in the new listAssignement and in the 
        **         first listAssignement).
        ** @param1: new matching list to fill
        */
        void                            _updateLists(std::vector<uintList>&);

        /*
        ** @brief: Reverse the order of match ids found in the first listAssignement to insert
        **         them next or prev the current one.
        ** @param1: current id
        ** @param2: position in the new matching list to insert
        ** @param3: position of the precedent id
        ** @param4: iterator pointing to the new matching list
        ** @param5: sequence of existing match ids to merge
        **
        */
        void                            _mergeIDs(uint, 
                                                  iteratorList&, 
                                                  iteratorList&, 
                                                  iteratorVector&, 
                                                  std::pair<Direction, uintList>&);

        /*
        ** @brief: Check if ids have already been added in the new list to return
        **          the current position in the vector and the current postion
        **          of the elem.
        **
        */
        const std::pair<iteratorVector,
                        iteratorList>   _foundElemNewList(const uint id, 
                                                          std::vector<uintList>&);
    };

#endif // !MANAGER_HH_
