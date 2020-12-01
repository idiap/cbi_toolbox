// Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
// Written by François Marelli <francois.marelli@idiap.ch>

// This file is part of CBI Toolbox.

// CBI Toolbox is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as
// published by the Free Software Foundation.

// CBI Toolbox is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with CBI Toolbox. If not, see <http://www.gnu.org/licenses/>.

#include "cradon.h"

#ifdef CUDA

bool is_cuda_available()
{
    auto dev_list = compatible_cuda_devices();
    return !dev_list.empty();
}

#else //CUDA

bool is_cuda_available()
{
    throw std::runtime_error("CUDA support is not installed.");
}

#endif //CUDA