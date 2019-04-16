//
// Created by fmarelli on 16/04/19.
//

#include "splineradon.h"

py::array_t<double> radon(py::array_t<double> image, double a){
//    auto array = image.unchecked<3>();

    auto array_info = image.request();

    auto output = py::array_t<uint16_t>(array_info);

    auto out_array = output.mutable_unchecked<3>();
    out_array(0,0,0) = (unsigned short int) a;

    return output;
}
