# include "ListAssignement.hh"

ListAssignement::ListAssignement(cv::Mat_<double>& hungarianMatrice, 
                                 const std::unordered_map<uint, uint>& indexToIdFrame) 
    : _hungarianMatrice(hungarianMatrice), _indexToIdFrame(indexToIdFrame)
{
}

ListAssignement::ListAssignement(const std::vector<uintList>& mergedList) 
    : _close(mergedList)
{
}

ListAssignement::~ListAssignement()
{
}

void                                    ListAssignement::sort()
{
    uintList freeIds;
    const uint nrows = static_cast<uint>(_hungarianMatrice.rows);
    const uint ncols = static_cast<uint>(_hungarianMatrice.cols);

    uintList openIds;
    for (uint idRow = 0; idRow < nrows; ++idRow)
    {
        const double* row = _hungarianMatrice.ptr<double>(idRow);
        const auto& itCol = std::find(row, row + ncols, 0);
        if (itCol == row + ncols)
            freeIds.push_back(idRow); // row with no match
        else
        {
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

const std::array<bool, 2>                ListAssignement::_areTheyAlreadyAdded(const uint row, const uint col)
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


const std::vector<uintList>&            ListAssignement::getCloseList()
{
    if (_close.size() == 1)
        _close[0].reverse();
    return _close;
}

const std::vector<uint>                 ListAssignement::getExtremityIdClose()
{
    for (auto& list : _close)
    {
        list.reverse();
        auto& idFront = list.front();
        auto& idBack = list.back();
        _extremityIdClose.push_back(idFront);
        _extremityIdClose.push_back(idBack);
    }
    return _extremityIdClose;
}

void                                    ListAssignement::updateIndexToIdFrame()
{
    for (auto& list : _open)
        _close.push_back(std::move(list));

    _open.clear();
    for (auto& idLists : _close)
        for (auto & id : idLists)
        {
            id = _indexToIdFrame.at(id);
        }
}

std::pair<Direction, uintList>          ListAssignement::findId(const uint id)
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
