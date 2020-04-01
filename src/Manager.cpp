#include "Manager.hh"

Manager::Manager()
{}

Manager::~Manager()
{}

const std::pair<iteratorVector, 
                iteratorList>    Manager::_foundElemNewList(const uint id,
                                                            std::vector<uintList>& newCloseList)
{
    iteratorList                end;
    for (auto& itList = newCloseList.begin(); itList != newCloseList.end(); ++itList)
    {
        auto& itId = std::find(itList->begin(), itList->end(), id);
        if (itId != itList->end())
            return std::make_pair(itList, itId);
    }
    return std::make_pair(newCloseList.end(), iteratorList());
}

void    Manager::_mergeIDs(uint id, 
                           iteratorList& posId, 
                           iteratorList& prevId, 
                           iteratorVector& itNewList, 
                           std::pair<Direction, uintList>& foundedInFirst)
{    
    if (prevId != itNewList->begin())
    {
         posId = std::next(prevId) == itNewList->end() ? std::next(prevId, 1) : itNewList->end();
        if (foundedInFirst.first == BACK)
            foundedInFirst.second.reverse();
    }
    else
    {
        posId = prevId; // insert left to prevId
        if (foundedInFirst.first == FRONT)
            foundedInFirst.second.reverse();
    }
    
    itNewList->insert(posId, foundedInFirst.second.begin(), foundedInFirst.second.end());
    posId = std::find(itNewList->begin(), itNewList->end(), id);
}

/*  REDUCE FONCTION */
void            Manager::_updateLists(std::vector<uintList>& newCloseList)
{
    const auto& firstList = _assignLists[0];
    const auto& sndCloseList = _assignLists[1]->getCloseList();
    auto        itNewList = iteratorVector();
    auto        posId = iteratorList();
    auto        prevId = iteratorList();
    bool        sameList = false;

    for (auto& list : sndCloseList)
    {
        for (auto& id : list)
        {
            auto& foundedInFirst = firstList->findId(id);
            const auto& foundedInNew = _foundElemNewList(id, newCloseList);
            /*  CUT THOSE CONDITIONS*/
            if (foundedInNew.first == newCloseList.end() && !sameList)
            {
                newCloseList.push_back(foundedInFirst.second);
                itNewList = std::prev(newCloseList.end(), 1);
                posId = std::find(itNewList->begin(), itNewList->end(), id);
                prevId = posId;
                sameList = true;
                continue;
            }
            else if (foundedInNew.first != newCloseList.end())
            { // 29 in
                posId = foundedInNew.second; // 4, 3 
                auto& listinNew = *foundedInNew.first;
                if (!sameList)
                    itNewList = foundedInNew.first;
                else if (std::find(listinNew.begin(), listinNew.end(), *prevId) == listinNew.end())
                {
                    foundedInFirst.first = (*prevId == itNewList->front()) ? FRONT : BACK;
                    foundedInFirst.second = std::move(*itNewList);
                    std::vector<uintList>::const_iterator itConst = itNewList;
                    itNewList = foundedInNew.first;
                    _mergeIDs(id, posId, posId, itNewList, foundedInFirst);
                    newCloseList.erase(itConst);
                }
                prevId = posId;
                sameList = true;
                continue;
            }

            /*          INSERT ID  AND ITS MATCHS  in same list*/
            uint idToAdd = foundedInFirst.first == FRONT ?
                           foundedInFirst.second.front() : foundedInFirst.second.back();
            posId = std::find(itNewList->begin(), itNewList->end(), idToAdd);
            if (posId == itNewList->end())
                _mergeIDs(id, posId, prevId, itNewList, foundedInFirst);
            prevId = posId;
        }
        sameList = false;
    }
}

void        Manager::launchMatching(cv::Mat_<double>& hungarianMatrice,
                                    const std::unordered_map<uint, uint>& indexToIdFrame)
{
    auto&   newMatching = std::make_unique<ListAssignement>(hungarianMatrice, indexToIdFrame);

    newMatching->sort();
    newMatching->updateIndexToIdFrame();

    this->_assignLists.push_back(std::move(newMatching));

    if (this->_assignLists.size() == 2)
    {
        std::vector<uintList> newList;
        this->_updateLists(newList);

        _assignLists.clear();
        _assignLists.push_back(std::make_unique<ListAssignement>(std::move(newList)));
    }
}

const std::vector<uint>     Manager::getExtremityIdClose()
{
    auto&                   list = std::prev(this->_assignLists.end(), 1);
    return list->get()->getExtremityIdClose();
}

const std::vector<uintList>&    Manager::getCloseList(const uint id) const
{
    return this->_assignLists[id]->getCloseList();
}

void                      Manager::setCloseList(const uint idList,
                                                const std::map<uint, std::vector<uint>>& idToErase,
                                                const std::vector<uint>& swap)
{
    this->_assignLists[idList]->setCloseList(idToErase, swap);
}
