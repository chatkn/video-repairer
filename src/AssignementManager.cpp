#include "AssignementManager.hh"

List::List(cv::Mat_<double>& hungarianMatrice, const std::unordered_map<uint, uint>& indexToIdFrame) :
    _hungarianMatrice(hungarianMatrice), _indexToIdFrame(indexToIdFrame)
{
}

List::List(const std::vector<uintList>& mergedList):
    _close(mergedList)
{
}

List::~List()
{
}

void                                    List::sort()
{
    uintList freeIds;
    const uint nrows = static_cast<uint>(_hungarianMatrice.rows);
    const uint ncols = static_cast<uint>(_hungarianMatrice.cols);
    std::cout << "[Assignement] mat size: " << nrows << "x"<< ncols << std::endl;

    uintList openIds;
    for (uint idRow = 0; idRow < nrows; ++idRow)
    {
        const double* row = _hungarianMatrice.ptr<double>(idRow);
        const auto& itCol = std::find(row, row + ncols, 0);
        if (itCol == row + ncols)
            freeIds.push_back(idRow); // row with no match
        else
        {// 4 9 0 8 7 9 8
            const uint idCol = static_cast<uint>(itCol - row);
            if (!_open.empty())
            {
                auto & rowColAdded = _areTheyAlreadyAdded(idRow, idCol);
                if (rowColAdded[0] || rowColAdded[1])
                    continue;
            }
            openIds.push_back(idRow);
            if (idCol != idRow)
                openIds.push_back(idCol);
        }
        _open.push_back(openIds);
        openIds.clear();
    }

}

const std::array<bool,2>                List::_areTheyAlreadyAdded(const uint row, const uint col)
{
    bool rowExist, colExist = false;
    for (auto& itList = _open.begin(); itList != _open.end(); ++itList)
    {
        const auto itrow = std::find(itList->begin(), itList->end(), row);
        rowExist = itrow != itList->end();
       
        auto itcol = std::find(itList->begin(), itList->end(), col);
        colExist = itcol != itList->end();
        
        if (rowExist && colExist) // add list to close list
        {   
            _close.push_back(std::move(*itList));
            std::vector<uintList>::const_iterator itConst = itList;
            itList = _open.erase(itConst);
            break;
        }

        if (rowExist && col != row)
        {// add col right to row
            itList->push_back(col);
            break;
        }

        if (colExist && row != col)
        {// add row left to col
            itcol = std::prev(itcol, 1);
            itList->insert(itcol, row);
            break;
        }
    }

    return { rowExist, colExist };
}


const std::vector<uintList>&            List::getCloseList() const
{
    return _close;
}

const std::vector<uint>                 List::getExtremityIdClose()
{
    for (const auto& list: _close)
    {
        auto& idFront = list.front();
        auto& idBack = list.back();
        _extremityIdClose.push_back(idFront);
        _extremityIdClose.push_back(idBack);
    }
    return _extremityIdClose;
}

void                                    List::updateIndexToIdFrame()
{
    std::cout << "OPEN Size: " << _open.size() << std::endl;
    for (auto& list : _open)
    {
        _close.push_back(std::move(list));
        std::cout << "list Size: " << list.size() << std::endl;
    }
    _open.clear();

    for (auto& idLists : _close)
        for (auto & id : idLists)
            id = _indexToIdFrame[id];


}

std::pair<Direction, uintList>          List::findId(const uint id)
{
    Direction direction = Direction::NONE;
    uintList nextToId;
    for (const auto& list : _close)
    {
        auto & itFound = std::find(list.begin(), list.end(), id);
        if (itFound != list.end())
        {
            direction = std::next(itFound, 1) == list.end() ? Direction::BACK : Direction::FRONT;
            nextToId = list;
        }
    }
    return std::make_pair(direction, nextToId);
}




Assignement::Assignement()
{
}

Assignement::~Assignement()
{
}

const std::pair<iteratorVector, iteratorList>    Assignement::_foundElemNewList(const uint id,
                                                                                  std::vector<uintList>& newCloseList)
{
    iteratorList    end;
    for (auto& itList = newCloseList.begin(); itList != newCloseList.end(); ++itList)
    {
        auto& itId = std::find(itList->begin(), itList->end(), id);
        if (itId != itList->end())
            return std::make_pair(itList, itId);
    }
    return std::make_pair(newCloseList.end(), iteratorList());
}

void                        Assignement::_mergeIDs(uint id, iteratorList& posId, 
                                                   iteratorList& prevId, iteratorVector& itNewList, 
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

void                        Assignement::_updateLists(std::vector<uintList>& newCloseList)
{
    const auto& firstList = _assignLists[0];
    const auto& sndCloseList = _assignLists[1]->getCloseList();
    
    auto itNewList = iteratorVector();
    auto posId = iteratorList();
    auto prevId = iteratorList();
    bool sameList = false;

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
            /*          INSERT ID  AND ITS COUPLES  in same list*/
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

void                        Assignement::launchMatching(cv::Mat_<double>& hungarianMatrice,
                                                        const std::unordered_map<uint, uint>& indexToIdFrame)
{
    auto& newMatching = std::make_unique<List>(hungarianMatrice, indexToIdFrame);

    newMatching->sort();
    newMatching->updateIndexToIdFrame();

    this->_assignLists.push_back(std::move(newMatching));

    if (this->_assignLists.size() == 2)
    {
        std::vector<uintList> newList;
        this->_updateLists(newList);

        while (!_assignLists.empty())
            _assignLists.pop_back();

        _assignLists.push_back(std::make_unique<List>(std::move(newList)));
    }
}

const std::vector<uint>     Assignement::getExtremityIdClose()
{
    auto& list = std::prev(this->_assignLists.end(), 1);

    return list->get()->getExtremityIdClose();
}

const std::vector<uintList>&    Assignement::getCloseList(const uint id) const
{
    return this->_assignLists[id]->getCloseList();
}
