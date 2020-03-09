//
//    rfnoc-hls-neuralnet: Vivado HLS code for neural-net building blocks
//
//    Copyright (C) 2017 EJ Kreinar
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <http://www.gnu.org/licenses/>.
//

#ifndef NNET_MERGE_H_
#define NNET_MERGE_H_

#include "nnet_common.h"
#include "hls_stream.h"
#include <math.h>

namespace nnet {

struct merge_config
{
    static const unsigned n_elem = 10;
};

struct concat_config {
    static const unsigned n_elem1_0 = 10;
    static const unsigned n_elem1_1 = 10;
    static const unsigned n_elem1_2 = 10;
    static const unsigned n_elem2_0 = 10;
    static const unsigned n_elem2_1 = 10;
    static const unsigned n_elem2_2 = 10;

    static const unsigned axis = -1;
};

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void add(
    input1_T data1[CONFIG_T::n_elem],
    input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
      #pragma HLS UNROLL
      res[ii] = data1[ii] + data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void subtract(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] - data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void multiply(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] * data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void average(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = data1[ii] * data2[ii] / (res_T) 2;
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void maximum(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = (data1[ii] > data2[ii]) ? data1[ii] : data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void minimum(
    input1_T data1[CONFIG_T::n_elem],
	input2_T data2[CONFIG_T::n_elem],
    res_T res[CONFIG_T::n_elem])
{
    for (int ii=0; ii<CONFIG_T::n_elem; ii++) {
        res[ii] = (data1[ii] < data2[ii]) ? data1[ii] : data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate1d(
    input1_T data1[CONFIG_T::n_elem1_0], 
	input2_T data2[CONFIG_T::n_elem2_0],
    res_T res[CONFIG_T::n_elem1_0 + CONFIG_T::n_elem2_0])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem2_0; ii++) {
        res[CONFIG_T::n_elem1_0 + ii] = data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_0(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1; ii++) {
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + ii] = data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d_1(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
        for (int jj=0; jj<CONFIG_T::n_elem1_1; jj++) {
            res[ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + jj] = data1[ii * CONFIG_T::n_elem1_1 + jj];
        }
        for (int jj=0; jj<CONFIG_T::n_elem2_1; jj++) {
            res[ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + CONFIG_T::n_elem1_1 + jj] = data2[ii * CONFIG_T::n_elem2_1 + jj];
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate2d(
    input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1])
{
    if (CONFIG_T::axis == 1 || CONFIG_T::axis == -1) {
        concatenate2d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate2d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_0(
input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2],
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2; ii++) {
        res[ii] = data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2; ii++) {
        res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + ii] = data2[ii];
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_1(
input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
        for (int jj=0; jj<CONFIG_T::n_elem1_1; jj++) {
            for (int kk=0; kk<CONFIG_T::n_elem1_2; kk++) {
                int res_idx = ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) * CONFIG_T::n_elem1_2
                            + jj * CONFIG_T::n_elem1_2
                            + kk;
                int data_idx = ii * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2
                             + jj * CONFIG_T::n_elem1_2
                             + kk;
                res[res_idx] = data1[data_idx];
            }
        }
        for (int jj=0; jj<CONFIG_T::n_elem2_1; jj++) {
            for (int kk=0; kk<CONFIG_T::n_elem2_2; kk++) {
                int res_idx = ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) * CONFIG_T::n_elem1_2
                            + (jj + CONFIG_T::n_elem1_1) * CONFIG_T::n_elem1_2
                            + kk;
                int data_idx = ii * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2
                             + jj * CONFIG_T::n_elem2_2
                             + kk;
                res[res_idx] = data2[data_idx];
            }
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d_2(
input1_T data1[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2], 
	input2_T data2[CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T res[CONFIG_T::n_elem1_0 * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_0 * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
        for (int jj=0; jj<CONFIG_T::n_elem1_1; jj++) {
            for (int kk=0; kk<CONFIG_T::n_elem1_2; kk++) {
                int res_idx = ii * CONFIG_T::n_elem1_1 * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2)
                            + jj * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2)
                            + kk;
                int data_idx = ii * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2
                             + jj * CONFIG_T::n_elem1_2
                             + kk;
                res[res_idx] = data1[data_idx];
            }
            for (int kk=0; kk<CONFIG_T::n_elem1_2; kk++) {
                res[ii * (CONFIG_T::n_elem1_1 + CONFIG_T::n_elem2_1) + CONFIG_T::n_elem1_1 + jj] = data1[ii * CONFIG_T::n_elem2_1 + jj];
                int res_idx = ii * CONFIG_T::n_elem1_1 * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2)
                            + jj * (CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2_2)
                            + kk + CONFIG_T::n_elem1_2;
                int data_idx = ii * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2
                             + jj * CONFIG_T::n_elem2_2
                             + kk;
                res[res_idx] = data2[data_idx];
            }
        }
    }
}

template<class input1_T, class input2_T, class res_T, typename CONFIG_T>
void concatenate3d(
    input1_T data1[CONFIG_T::n_elem1[0] * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2], 
	input2_T data2[CONFIG_T::n_elem2[0] * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2],
    res_T res[CONFIG_T::n_elem1[0] * CONFIG_T::n_elem1_1 * CONFIG_T::n_elem1_2 + CONFIG_T::n_elem2[0] * CONFIG_T::n_elem2_1 * CONFIG_T::n_elem2_2])
{
    if (CONFIG_T::axis == 2 || CONFIG_T::axis == -1) {
        concatenate3d_2<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else if (CONFIG_T::axis == 1) {
        concatenate3d_1<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    } else {
        concatenate3d_0<input1_T, input2_T, res_T, CONFIG_T>(data1, data2, res);
    }
}

template<class input_T, typename CONFIG_T>
void merge1d_128(
    input_T data0[CONFIG_T::n_elem1_0],
    input_T data1[CONFIG_T::n_elem1_0],
    input_T data2[CONFIG_T::n_elem1_0],
    input_T data3[CONFIG_T::n_elem1_0],
    input_T data4[CONFIG_T::n_elem1_0],
    input_T data5[CONFIG_T::n_elem1_0],
    input_T data6[CONFIG_T::n_elem1_0],
    input_T data7[CONFIG_T::n_elem1_0],
    input_T data8[CONFIG_T::n_elem1_0],
    input_T data9[CONFIG_T::n_elem1_0],
    input_T data10[CONFIG_T::n_elem1_0],
    input_T data11[CONFIG_T::n_elem1_0],
    input_T data12[CONFIG_T::n_elem1_0],
    input_T data13[CONFIG_T::n_elem1_0],
    input_T data14[CONFIG_T::n_elem1_0],
    input_T data15[CONFIG_T::n_elem1_0],
    input_T data16[CONFIG_T::n_elem1_0],
    input_T data17[CONFIG_T::n_elem1_0],
    input_T data18[CONFIG_T::n_elem1_0],
    input_T data19[CONFIG_T::n_elem1_0],
    input_T data20[CONFIG_T::n_elem1_0],
    input_T data21[CONFIG_T::n_elem1_0],
    input_T data22[CONFIG_T::n_elem1_0],
    input_T data23[CONFIG_T::n_elem1_0],
    input_T data24[CONFIG_T::n_elem1_0],
    input_T data25[CONFIG_T::n_elem1_0],
    input_T data26[CONFIG_T::n_elem1_0],
    input_T data27[CONFIG_T::n_elem1_0],
    input_T data28[CONFIG_T::n_elem1_0],
    input_T data29[CONFIG_T::n_elem1_0],
    input_T data30[CONFIG_T::n_elem1_0],
    input_T data31[CONFIG_T::n_elem1_0],
    input_T data32[CONFIG_T::n_elem1_0],
    input_T data33[CONFIG_T::n_elem1_0],
    input_T data34[CONFIG_T::n_elem1_0],
    input_T data35[CONFIG_T::n_elem1_0],
    input_T data36[CONFIG_T::n_elem1_0],
    input_T data37[CONFIG_T::n_elem1_0],
    input_T data38[CONFIG_T::n_elem1_0],
    input_T data39[CONFIG_T::n_elem1_0],
    input_T data40[CONFIG_T::n_elem1_0],
    input_T data41[CONFIG_T::n_elem1_0],
    input_T data42[CONFIG_T::n_elem1_0],
    input_T data43[CONFIG_T::n_elem1_0],
    input_T data44[CONFIG_T::n_elem1_0],
    input_T data45[CONFIG_T::n_elem1_0],
    input_T data46[CONFIG_T::n_elem1_0],
    input_T data47[CONFIG_T::n_elem1_0],
    input_T data48[CONFIG_T::n_elem1_0],
    input_T data49[CONFIG_T::n_elem1_0],
    input_T data50[CONFIG_T::n_elem1_0],
    input_T data51[CONFIG_T::n_elem1_0],
    input_T data52[CONFIG_T::n_elem1_0],
    input_T data53[CONFIG_T::n_elem1_0],
    input_T data54[CONFIG_T::n_elem1_0],
    input_T data55[CONFIG_T::n_elem1_0],
    input_T data56[CONFIG_T::n_elem1_0],
    input_T data57[CONFIG_T::n_elem1_0],
    input_T data58[CONFIG_T::n_elem1_0],
    input_T data59[CONFIG_T::n_elem1_0],
    input_T data60[CONFIG_T::n_elem1_0],
    input_T data61[CONFIG_T::n_elem1_0],
    input_T data62[CONFIG_T::n_elem1_0],
    input_T data63[CONFIG_T::n_elem1_0],
    input_T data64[CONFIG_T::n_elem1_0],
    input_T data65[CONFIG_T::n_elem1_0],
    input_T data66[CONFIG_T::n_elem1_0],
    input_T data67[CONFIG_T::n_elem1_0],
    input_T data68[CONFIG_T::n_elem1_0],
    input_T data69[CONFIG_T::n_elem1_0],
    input_T data70[CONFIG_T::n_elem1_0],
    input_T data71[CONFIG_T::n_elem1_0],
    input_T data72[CONFIG_T::n_elem1_0],
    input_T data73[CONFIG_T::n_elem1_0],
    input_T data74[CONFIG_T::n_elem1_0],
    input_T data75[CONFIG_T::n_elem1_0],
    input_T data76[CONFIG_T::n_elem1_0],
    input_T data77[CONFIG_T::n_elem1_0],
    input_T data78[CONFIG_T::n_elem1_0],
    input_T data79[CONFIG_T::n_elem1_0],
    input_T data80[CONFIG_T::n_elem1_0],
    input_T data81[CONFIG_T::n_elem1_0],
    input_T data82[CONFIG_T::n_elem1_0],
    input_T data83[CONFIG_T::n_elem1_0],
    input_T data84[CONFIG_T::n_elem1_0],
    input_T data85[CONFIG_T::n_elem1_0],
    input_T data86[CONFIG_T::n_elem1_0],
    input_T data87[CONFIG_T::n_elem1_0],
    input_T data88[CONFIG_T::n_elem1_0],
    input_T data89[CONFIG_T::n_elem1_0],
    input_T data90[CONFIG_T::n_elem1_0],
    input_T data91[CONFIG_T::n_elem1_0],
    input_T data92[CONFIG_T::n_elem1_0],
    input_T data93[CONFIG_T::n_elem1_0],
    input_T data94[CONFIG_T::n_elem1_0],
    input_T data95[CONFIG_T::n_elem1_0],
    input_T data96[CONFIG_T::n_elem1_0],
    input_T data97[CONFIG_T::n_elem1_0],
    input_T data98[CONFIG_T::n_elem1_0],
    input_T data99[CONFIG_T::n_elem1_0],
    input_T data100[CONFIG_T::n_elem1_0],
    input_T data101[CONFIG_T::n_elem1_0],
    input_T data102[CONFIG_T::n_elem1_0],
    input_T data103[CONFIG_T::n_elem1_0],
    input_T data104[CONFIG_T::n_elem1_0],
    input_T data105[CONFIG_T::n_elem1_0],
    input_T data106[CONFIG_T::n_elem1_0],
    input_T data107[CONFIG_T::n_elem1_0],
    input_T data108[CONFIG_T::n_elem1_0],
    input_T data109[CONFIG_T::n_elem1_0],
    input_T data110[CONFIG_T::n_elem1_0],
    input_T data111[CONFIG_T::n_elem1_0],
    input_T data112[CONFIG_T::n_elem1_0],
    input_T data113[CONFIG_T::n_elem1_0],
    input_T data114[CONFIG_T::n_elem1_0],
    input_T data115[CONFIG_T::n_elem1_0],
    input_T data116[CONFIG_T::n_elem1_0],
    input_T data117[CONFIG_T::n_elem1_0],
    input_T data118[CONFIG_T::n_elem1_0],
    input_T data119[CONFIG_T::n_elem1_0],
    input_T data120[CONFIG_T::n_elem1_0],
    input_T data121[CONFIG_T::n_elem1_0],
    input_T data122[CONFIG_T::n_elem1_0],
    input_T data123[CONFIG_T::n_elem1_0],
    input_T data124[CONFIG_T::n_elem1_0],
    input_T data125[CONFIG_T::n_elem1_0],
    input_T data126[CONFIG_T::n_elem1_0],
    input_T data127[CONFIG_T::n_elem1_0],
    input_T res[CONFIG_T::n_elem2_0])
{
    input_T acc[CONFIG_T::n_elem2_0];
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] = data0[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data2[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data3[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data4[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data5[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data6[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data7[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data8[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data9[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data10[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data11[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data12[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data13[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data14[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data15[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data16[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data17[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data18[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data19[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data20[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data21[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data22[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data23[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data24[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data25[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data26[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data27[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data28[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data29[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data30[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data31[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data32[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data33[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data34[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data35[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data36[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data37[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data38[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data39[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data40[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data41[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data42[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data43[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data44[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data45[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data46[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data47[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data48[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data49[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data50[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data51[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data52[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data53[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data54[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data55[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data56[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data57[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data58[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data59[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data60[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data61[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data62[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data63[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data64[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data65[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data66[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data67[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data68[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data69[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data70[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data71[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data72[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data73[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data74[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data75[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data76[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data77[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data78[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data79[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data80[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data81[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data82[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data83[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data84[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data85[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data86[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data87[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data88[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data89[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data90[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data91[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data92[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data93[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data94[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data95[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data96[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data97[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data98[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data99[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data100[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data101[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data102[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data103[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data104[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data105[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data106[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data107[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data108[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data109[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data110[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data111[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data112[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data113[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data114[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data115[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data116[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data117[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data118[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data119[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data120[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data121[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data122[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data123[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data124[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data125[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data126[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        acc[ii] += data127[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[ii] = acc[ii];
    }
}

template<class input_T, typename CONFIG_T>
void concatenate1d_16(
    input_T data0[CONFIG_T::n_elem1_0],
    input_T data1[CONFIG_T::n_elem1_0],
    input_T data2[CONFIG_T::n_elem1_0],
    input_T data3[CONFIG_T::n_elem1_0],
    input_T data4[CONFIG_T::n_elem1_0],
    input_T data5[CONFIG_T::n_elem1_0],
    input_T data6[CONFIG_T::n_elem1_0],
    input_T data7[CONFIG_T::n_elem1_0],
    input_T data8[CONFIG_T::n_elem1_0],
    input_T data9[CONFIG_T::n_elem1_0],
    input_T data10[CONFIG_T::n_elem1_0],
    input_T data11[CONFIG_T::n_elem1_0],
    input_T data12[CONFIG_T::n_elem1_0],
    input_T data13[CONFIG_T::n_elem1_0],
    input_T data14[CONFIG_T::n_elem1_0],
    input_T data15[CONFIG_T::n_elem1_0],
    input_T res[CONFIG_T::n_elem2_0]) { 

    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(0*(CONFIG_T::n_elem1_0)) + ii] = data0[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(1*(CONFIG_T::n_elem1_0)) + ii] = data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(2*(CONFIG_T::n_elem1_0)) + ii] = data2[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(3*(CONFIG_T::n_elem1_0)) + ii] = data3[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(4*(CONFIG_T::n_elem1_0)) + ii] = data4[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(5*(CONFIG_T::n_elem1_0)) + ii] = data5[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(6*(CONFIG_T::n_elem1_0)) + ii] = data6[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(7*(CONFIG_T::n_elem1_0)) + ii] = data7[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(8*(CONFIG_T::n_elem1_0)) + ii] = data8[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(9*(CONFIG_T::n_elem1_0)) + ii] = data9[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(10*(CONFIG_T::n_elem1_0)) + ii] = data10[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(11*(CONFIG_T::n_elem1_0)) + ii] = data11[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(12*(CONFIG_T::n_elem1_0)) + ii] = data12[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(13*(CONFIG_T::n_elem1_0)) + ii] = data13[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(14*(CONFIG_T::n_elem1_0)) + ii] = data14[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(15*(CONFIG_T::n_elem1_0)) + ii] = data15[ii];
    }
}

template<class input_T, typename CONFIG_T>
void concatenate1d_128(
    input_T data0[CONFIG_T::n_elem1_0],
    input_T data1[CONFIG_T::n_elem1_0],
    input_T data2[CONFIG_T::n_elem1_0],
    input_T data3[CONFIG_T::n_elem1_0],
    input_T data4[CONFIG_T::n_elem1_0],
    input_T data5[CONFIG_T::n_elem1_0],
    input_T data6[CONFIG_T::n_elem1_0],
    input_T data7[CONFIG_T::n_elem1_0],
    input_T data8[CONFIG_T::n_elem1_0],
    input_T data9[CONFIG_T::n_elem1_0],
    input_T data10[CONFIG_T::n_elem1_0],
    input_T data11[CONFIG_T::n_elem1_0],
    input_T data12[CONFIG_T::n_elem1_0],
    input_T data13[CONFIG_T::n_elem1_0],
    input_T data14[CONFIG_T::n_elem1_0],
    input_T data15[CONFIG_T::n_elem1_0],
    input_T data16[CONFIG_T::n_elem1_0],
    input_T data17[CONFIG_T::n_elem1_0],
    input_T data18[CONFIG_T::n_elem1_0],
    input_T data19[CONFIG_T::n_elem1_0],
    input_T data20[CONFIG_T::n_elem1_0],
    input_T data21[CONFIG_T::n_elem1_0],
    input_T data22[CONFIG_T::n_elem1_0],
    input_T data23[CONFIG_T::n_elem1_0],
    input_T data24[CONFIG_T::n_elem1_0],
    input_T data25[CONFIG_T::n_elem1_0],
    input_T data26[CONFIG_T::n_elem1_0],
    input_T data27[CONFIG_T::n_elem1_0],
    input_T data28[CONFIG_T::n_elem1_0],
    input_T data29[CONFIG_T::n_elem1_0],
    input_T data30[CONFIG_T::n_elem1_0],
    input_T data31[CONFIG_T::n_elem1_0],
    input_T data32[CONFIG_T::n_elem1_0],
    input_T data33[CONFIG_T::n_elem1_0],
    input_T data34[CONFIG_T::n_elem1_0],
    input_T data35[CONFIG_T::n_elem1_0],
    input_T data36[CONFIG_T::n_elem1_0],
    input_T data37[CONFIG_T::n_elem1_0],
    input_T data38[CONFIG_T::n_elem1_0],
    input_T data39[CONFIG_T::n_elem1_0],
    input_T data40[CONFIG_T::n_elem1_0],
    input_T data41[CONFIG_T::n_elem1_0],
    input_T data42[CONFIG_T::n_elem1_0],
    input_T data43[CONFIG_T::n_elem1_0],
    input_T data44[CONFIG_T::n_elem1_0],
    input_T data45[CONFIG_T::n_elem1_0],
    input_T data46[CONFIG_T::n_elem1_0],
    input_T data47[CONFIG_T::n_elem1_0],
    input_T data48[CONFIG_T::n_elem1_0],
    input_T data49[CONFIG_T::n_elem1_0],
    input_T data50[CONFIG_T::n_elem1_0],
    input_T data51[CONFIG_T::n_elem1_0],
    input_T data52[CONFIG_T::n_elem1_0],
    input_T data53[CONFIG_T::n_elem1_0],
    input_T data54[CONFIG_T::n_elem1_0],
    input_T data55[CONFIG_T::n_elem1_0],
    input_T data56[CONFIG_T::n_elem1_0],
    input_T data57[CONFIG_T::n_elem1_0],
    input_T data58[CONFIG_T::n_elem1_0],
    input_T data59[CONFIG_T::n_elem1_0],
    input_T data60[CONFIG_T::n_elem1_0],
    input_T data61[CONFIG_T::n_elem1_0],
    input_T data62[CONFIG_T::n_elem1_0],
    input_T data63[CONFIG_T::n_elem1_0],
    input_T data64[CONFIG_T::n_elem1_0],
    input_T data65[CONFIG_T::n_elem1_0],
    input_T data66[CONFIG_T::n_elem1_0],
    input_T data67[CONFIG_T::n_elem1_0],
    input_T data68[CONFIG_T::n_elem1_0],
    input_T data69[CONFIG_T::n_elem1_0],
    input_T data70[CONFIG_T::n_elem1_0],
    input_T data71[CONFIG_T::n_elem1_0],
    input_T data72[CONFIG_T::n_elem1_0],
    input_T data73[CONFIG_T::n_elem1_0],
    input_T data74[CONFIG_T::n_elem1_0],
    input_T data75[CONFIG_T::n_elem1_0],
    input_T data76[CONFIG_T::n_elem1_0],
    input_T data77[CONFIG_T::n_elem1_0],
    input_T data78[CONFIG_T::n_elem1_0],
    input_T data79[CONFIG_T::n_elem1_0],
    input_T data80[CONFIG_T::n_elem1_0],
    input_T data81[CONFIG_T::n_elem1_0],
    input_T data82[CONFIG_T::n_elem1_0],
    input_T data83[CONFIG_T::n_elem1_0],
    input_T data84[CONFIG_T::n_elem1_0],
    input_T data85[CONFIG_T::n_elem1_0],
    input_T data86[CONFIG_T::n_elem1_0],
    input_T data87[CONFIG_T::n_elem1_0],
    input_T data88[CONFIG_T::n_elem1_0],
    input_T data89[CONFIG_T::n_elem1_0],
    input_T data90[CONFIG_T::n_elem1_0],
    input_T data91[CONFIG_T::n_elem1_0],
    input_T data92[CONFIG_T::n_elem1_0],
    input_T data93[CONFIG_T::n_elem1_0],
    input_T data94[CONFIG_T::n_elem1_0],
    input_T data95[CONFIG_T::n_elem1_0],
    input_T data96[CONFIG_T::n_elem1_0],
    input_T data97[CONFIG_T::n_elem1_0],
    input_T data98[CONFIG_T::n_elem1_0],
    input_T data99[CONFIG_T::n_elem1_0],
    input_T data100[CONFIG_T::n_elem1_0],
    input_T data101[CONFIG_T::n_elem1_0],
    input_T data102[CONFIG_T::n_elem1_0],
    input_T data103[CONFIG_T::n_elem1_0],
    input_T data104[CONFIG_T::n_elem1_0],
    input_T data105[CONFIG_T::n_elem1_0],
    input_T data106[CONFIG_T::n_elem1_0],
    input_T data107[CONFIG_T::n_elem1_0],
    input_T data108[CONFIG_T::n_elem1_0],
    input_T data109[CONFIG_T::n_elem1_0],
    input_T data110[CONFIG_T::n_elem1_0],
    input_T data111[CONFIG_T::n_elem1_0],
    input_T data112[CONFIG_T::n_elem1_0],
    input_T data113[CONFIG_T::n_elem1_0],
    input_T data114[CONFIG_T::n_elem1_0],
    input_T data115[CONFIG_T::n_elem1_0],
    input_T data116[CONFIG_T::n_elem1_0],
    input_T data117[CONFIG_T::n_elem1_0],
    input_T data118[CONFIG_T::n_elem1_0],
    input_T data119[CONFIG_T::n_elem1_0],
    input_T data120[CONFIG_T::n_elem1_0],
    input_T data121[CONFIG_T::n_elem1_0],
    input_T data122[CONFIG_T::n_elem1_0],
    input_T data123[CONFIG_T::n_elem1_0],
    input_T data124[CONFIG_T::n_elem1_0],
    input_T data125[CONFIG_T::n_elem1_0],
    input_T data126[CONFIG_T::n_elem1_0],
    input_T data127[CONFIG_T::n_elem1_0],
    input_T res[CONFIG_T::n_elem2_0])
{
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(0*(CONFIG_T::n_elem1_0)) + ii] = data0[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(1*(CONFIG_T::n_elem1_0)) + ii] = data1[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(2*(CONFIG_T::n_elem1_0)) + ii] = data2[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(3*(CONFIG_T::n_elem1_0)) + ii] = data3[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(4*(CONFIG_T::n_elem1_0)) + ii] = data4[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(5*(CONFIG_T::n_elem1_0)) + ii] = data5[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(6*(CONFIG_T::n_elem1_0)) + ii] = data6[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(7*(CONFIG_T::n_elem1_0)) + ii] = data7[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(8*(CONFIG_T::n_elem1_0)) + ii] = data8[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(9*(CONFIG_T::n_elem1_0)) + ii] = data9[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(10*(CONFIG_T::n_elem1_0)) + ii] = data10[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(11*(CONFIG_T::n_elem1_0)) + ii] = data11[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(12*(CONFIG_T::n_elem1_0)) + ii] = data12[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(13*(CONFIG_T::n_elem1_0)) + ii] = data13[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(14*(CONFIG_T::n_elem1_0)) + ii] = data14[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(15*(CONFIG_T::n_elem1_0)) + ii] = data15[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(16*(CONFIG_T::n_elem1_0)) + ii] = data16[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(17*(CONFIG_T::n_elem1_0)) + ii] = data17[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(18*(CONFIG_T::n_elem1_0)) + ii] = data18[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(19*(CONFIG_T::n_elem1_0)) + ii] = data19[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(20*(CONFIG_T::n_elem1_0)) + ii] = data20[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(21*(CONFIG_T::n_elem1_0)) + ii] = data21[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(22*(CONFIG_T::n_elem1_0)) + ii] = data22[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(23*(CONFIG_T::n_elem1_0)) + ii] = data23[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(24*(CONFIG_T::n_elem1_0)) + ii] = data24[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(25*(CONFIG_T::n_elem1_0)) + ii] = data25[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(26*(CONFIG_T::n_elem1_0)) + ii] = data26[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(27*(CONFIG_T::n_elem1_0)) + ii] = data27[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(28*(CONFIG_T::n_elem1_0)) + ii] = data28[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(29*(CONFIG_T::n_elem1_0)) + ii] = data29[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(30*(CONFIG_T::n_elem1_0)) + ii] = data30[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(31*(CONFIG_T::n_elem1_0)) + ii] = data31[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(32*(CONFIG_T::n_elem1_0)) + ii] = data32[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(33*(CONFIG_T::n_elem1_0)) + ii] = data33[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(34*(CONFIG_T::n_elem1_0)) + ii] = data34[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(35*(CONFIG_T::n_elem1_0)) + ii] = data35[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(36*(CONFIG_T::n_elem1_0)) + ii] = data36[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(37*(CONFIG_T::n_elem1_0)) + ii] = data37[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(38*(CONFIG_T::n_elem1_0)) + ii] = data38[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(39*(CONFIG_T::n_elem1_0)) + ii] = data39[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(40*(CONFIG_T::n_elem1_0)) + ii] = data40[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(41*(CONFIG_T::n_elem1_0)) + ii] = data41[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(42*(CONFIG_T::n_elem1_0)) + ii] = data42[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(43*(CONFIG_T::n_elem1_0)) + ii] = data43[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(44*(CONFIG_T::n_elem1_0)) + ii] = data44[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(45*(CONFIG_T::n_elem1_0)) + ii] = data45[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(46*(CONFIG_T::n_elem1_0)) + ii] = data46[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(47*(CONFIG_T::n_elem1_0)) + ii] = data47[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(48*(CONFIG_T::n_elem1_0)) + ii] = data48[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(49*(CONFIG_T::n_elem1_0)) + ii] = data49[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(50*(CONFIG_T::n_elem1_0)) + ii] = data50[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(51*(CONFIG_T::n_elem1_0)) + ii] = data51[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(52*(CONFIG_T::n_elem1_0)) + ii] = data52[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(53*(CONFIG_T::n_elem1_0)) + ii] = data53[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(54*(CONFIG_T::n_elem1_0)) + ii] = data54[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(55*(CONFIG_T::n_elem1_0)) + ii] = data55[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(56*(CONFIG_T::n_elem1_0)) + ii] = data56[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(57*(CONFIG_T::n_elem1_0)) + ii] = data57[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(58*(CONFIG_T::n_elem1_0)) + ii] = data58[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(59*(CONFIG_T::n_elem1_0)) + ii] = data59[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(60*(CONFIG_T::n_elem1_0)) + ii] = data60[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(61*(CONFIG_T::n_elem1_0)) + ii] = data61[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(62*(CONFIG_T::n_elem1_0)) + ii] = data62[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(63*(CONFIG_T::n_elem1_0)) + ii] = data63[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(64*(CONFIG_T::n_elem1_0)) + ii] = data64[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(65*(CONFIG_T::n_elem1_0)) + ii] = data65[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(66*(CONFIG_T::n_elem1_0)) + ii] = data66[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(67*(CONFIG_T::n_elem1_0)) + ii] = data67[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(68*(CONFIG_T::n_elem1_0)) + ii] = data68[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(69*(CONFIG_T::n_elem1_0)) + ii] = data69[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(70*(CONFIG_T::n_elem1_0)) + ii] = data70[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(71*(CONFIG_T::n_elem1_0)) + ii] = data71[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(72*(CONFIG_T::n_elem1_0)) + ii] = data72[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(73*(CONFIG_T::n_elem1_0)) + ii] = data73[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(74*(CONFIG_T::n_elem1_0)) + ii] = data74[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(75*(CONFIG_T::n_elem1_0)) + ii] = data75[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(76*(CONFIG_T::n_elem1_0)) + ii] = data76[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(77*(CONFIG_T::n_elem1_0)) + ii] = data77[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(78*(CONFIG_T::n_elem1_0)) + ii] = data78[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(79*(CONFIG_T::n_elem1_0)) + ii] = data79[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(80*(CONFIG_T::n_elem1_0)) + ii] = data80[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(81*(CONFIG_T::n_elem1_0)) + ii] = data81[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(82*(CONFIG_T::n_elem1_0)) + ii] = data82[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(83*(CONFIG_T::n_elem1_0)) + ii] = data83[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(84*(CONFIG_T::n_elem1_0)) + ii] = data84[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(85*(CONFIG_T::n_elem1_0)) + ii] = data85[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(86*(CONFIG_T::n_elem1_0)) + ii] = data86[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(87*(CONFIG_T::n_elem1_0)) + ii] = data87[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(88*(CONFIG_T::n_elem1_0)) + ii] = data88[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(89*(CONFIG_T::n_elem1_0)) + ii] = data89[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(90*(CONFIG_T::n_elem1_0)) + ii] = data90[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(91*(CONFIG_T::n_elem1_0)) + ii] = data91[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(92*(CONFIG_T::n_elem1_0)) + ii] = data92[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(93*(CONFIG_T::n_elem1_0)) + ii] = data93[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(94*(CONFIG_T::n_elem1_0)) + ii] = data94[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(95*(CONFIG_T::n_elem1_0)) + ii] = data95[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(96*(CONFIG_T::n_elem1_0)) + ii] = data96[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(97*(CONFIG_T::n_elem1_0)) + ii] = data97[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(98*(CONFIG_T::n_elem1_0)) + ii] = data98[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(99*(CONFIG_T::n_elem1_0)) + ii] = data99[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(100*(CONFIG_T::n_elem1_0)) + ii] = data100[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(101*(CONFIG_T::n_elem1_0)) + ii] = data101[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(102*(CONFIG_T::n_elem1_0)) + ii] = data102[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(103*(CONFIG_T::n_elem1_0)) + ii] = data103[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(104*(CONFIG_T::n_elem1_0)) + ii] = data104[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(105*(CONFIG_T::n_elem1_0)) + ii] = data105[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(106*(CONFIG_T::n_elem1_0)) + ii] = data106[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(107*(CONFIG_T::n_elem1_0)) + ii] = data107[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(108*(CONFIG_T::n_elem1_0)) + ii] = data108[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(109*(CONFIG_T::n_elem1_0)) + ii] = data109[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(110*(CONFIG_T::n_elem1_0)) + ii] = data110[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(111*(CONFIG_T::n_elem1_0)) + ii] = data111[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(112*(CONFIG_T::n_elem1_0)) + ii] = data112[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(113*(CONFIG_T::n_elem1_0)) + ii] = data113[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(114*(CONFIG_T::n_elem1_0)) + ii] = data114[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(115*(CONFIG_T::n_elem1_0)) + ii] = data115[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(116*(CONFIG_T::n_elem1_0)) + ii] = data116[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(117*(CONFIG_T::n_elem1_0)) + ii] = data117[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(118*(CONFIG_T::n_elem1_0)) + ii] = data118[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(119*(CONFIG_T::n_elem1_0)) + ii] = data119[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(120*(CONFIG_T::n_elem1_0)) + ii] = data120[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(121*(CONFIG_T::n_elem1_0)) + ii] = data121[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(122*(CONFIG_T::n_elem1_0)) + ii] = data122[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(123*(CONFIG_T::n_elem1_0)) + ii] = data123[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(124*(CONFIG_T::n_elem1_0)) + ii] = data124[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(125*(CONFIG_T::n_elem1_0)) + ii] = data125[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(126*(CONFIG_T::n_elem1_0)) + ii] = data126[ii];
    }
    for (int ii=0; ii<CONFIG_T::n_elem1_0; ii++) {
    #pragma HLS LOOP UNROLL
        res[(127*(CONFIG_T::n_elem1_0)) + ii] = data127[ii];
    }
}


}
#endif