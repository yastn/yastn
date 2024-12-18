/*
# Copyright 2024 The YASTN Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
*/

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#include <omp.h>

#if defined(_MSC_VER)
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

#define MAX_ORDER 32

// temp = newdata[slice(*sln)].reshape(Dn)
// temp[slcs] = data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)
DLLEXPORT
int tm_worker(double *newdata, int sln0, int len_Dn, int *Dn, int *Dslc0,
              double *data, int slo0, int *Do, int *order, int *Drsh, int len_Do, int itemsize)
{
    int iR[MAX_ORDER], iL[MAX_ORDER];
    int strideL[MAX_ORDER], strideR[MAX_ORDER], strideD[MAX_ORDER];
    int addR[MAX_ORDER], addL[MAX_ORDER];
    int Dt[MAX_ORDER];

    int i, k, m, idxR, idxL, p;

    if (len_Do > MAX_ORDER || len_Dn > MAX_ORDER)
    {
        printf("tm_worker error: len_Do > 32 || len_Dn > 32 not implemented.\n");
        return -1;
    }
    if ((itemsize != 8) && (itemsize != 16))
    {
        printf("tm_worker error: itemsize %d not implemented.\n", itemsize);
        return -1;
    }

    for (k = 0; k < len_Do; k++)
    {
        Dt[k] = Do[order[k]];
    }

    p = itemsize;
    for (k = 0; k < len_Do; k++)
    {
        i = len_Do - k - 1;
        strideD[i] = p;
        p *= Do[i];
    }

    for (k = 0; k < len_Do; k++)
    {
        strideR[k] = strideD[order[k]];
    }

    p = itemsize;
    for (m = 0; m < len_Dn; m++)
    {
        i = len_Dn - m - 1;
        strideL[i] = p;
        p *= Dn[i];
    }

    addR[0] = 0;
    for (k = 1; k < len_Do; k++)
    {
        addR[k] = -Dt[k] * strideR[k] + strideR[k - 1];
    }

    addL[0] = 0;
    for (m = 1; m < len_Dn; m++)
    {
        addL[m] = -Drsh[m] * strideL[m] + strideL[m - 1];
    }

    // starting indexes for Left and Right-side loops
    idxL = 0;
    idxR = 0;
    for (k = 0; k < len_Do; k++)
        iR[k] = Dt[k];
    for (m = 0; m < len_Dn; m++)
        iL[m] = Drsh[m];

    int slcs = 0;
    for (m = 0; m < len_Dn; m++)
    {
        slcs = (slcs * Dn[m]) + Dslc0[m];
    }

    char *data_slice = (char *)data + itemsize * slo0;
    char *newdata_slice = (char *)newdata + itemsize * (sln0 + slcs);

    // loop over elements
    k = 1; // dummy value for start
    int strideR1 = strideR[len_Do - 1];
    int strideL1 = strideL[len_Dn - 1];

    while (k >= 0)
    {
        if (itemsize == 8)
            memcpy(newdata_slice + idxL, data_slice + idxR, 8);
        else
            memcpy(newdata_slice + idxL, data_slice + idxR, 16);

        // counter idxR is changed during looping over Do.transpose(order)
        k = len_Do - 1;
        idxR += strideR1;
        iR[k]--;
        while (iR[k] == 0)
        {
            iR[k] = Dt[k];
            idxR += addR[k];
            k--;
            if (k < 0)
                break;
            iR[k]--;
        }
        // simultaneously
        // counter idxL is changed during looping over Drsh, but strides from Dn
        m = len_Dn - 1;
        idxL += strideL1;
        iL[m]--;
        while (iL[m] == 0)
        {
            iL[m] = Drsh[m];
            idxL += addL[m];
            m--;
            if (m < 0)
                break;
            iL[m]--;
        }
    }
    return 0;
}

// this version works correctly, but includes some division/mutliplication which makes it a bit slower
// temp = newdata[slice(*sln)].reshape(Dn)
// temp[slcs] = data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)
DLLEXPORT
int tm_worker_beta(double *newdata, int sln0, int n_Dn, int *Dn, int *Dslc0,
                   double *data, int slo0, int *Do, int *order, int *Drsh, int n_order, int itemsize)
{
    int iDo[MAX_ORDER];      // indexes of Do array that are used to loop
    int strideDt[MAX_ORDER]; // stride of Do.transpose(order) array
    int strideDrsh[MAX_ORDER];
    int strideDn[MAX_ORDER];

    int j, k, idx_Do, idx_Dt, nDt, nDn, nDrsh, slcs, stride, shift, idx, idiv;

    if (n_order > MAX_ORDER || n_Dn > MAX_ORDER)
    {
        printf("tm_worker error: n_order > MAX_ORDER=32 not implemented.\n");
        return -1;
    }
    if ((itemsize != 8) && (itemsize != 16))
    {
        printf("tm_worker error: (itemsize != 8) && (itemsize != 16) not implemented.\n");
        return -1;
    }

    nDt = 1;
    for (j = 0; j < n_order; j++)
    {
        k = order[n_order - j - 1];
        strideDt[k] = nDt;
        nDt *= Do[k];
        iDo[j] = 0;
    }

    nDn = 1;
    nDrsh = 1;
    for (j = 0; j < n_Dn; j++)
    {
        k = n_Dn - j - 1;
        strideDn[k] = nDn;
        nDn *= Dn[k];
        strideDrsh[k] = nDrsh;
        nDrsh *= Drsh[k];
    }

    slcs = 0;
    for (j = 0; j < n_Dn; j++)
    {
        slcs = (slcs * Dn[j]) + Dslc0[j];
    }

    double *dataslo;
    double *newdataslc;
    if (itemsize == 8)
    {
        dataslo = data + slo0;
        newdataslc = newdata + sln0 + slcs;
    }
    else
    { // itemsize==16
        dataslo = data + 2 * slo0;
        newdataslc = newdata + 2 * (sln0 + slcs);
    }

    idx_Do = 0;
    idx_Dt = 0;
    shift = 0;
    idx = 0;

    // loop over elements
    k = 1; // dummy value for start
    while (k >= 0)
    {
        shift = 0;
        idx = idx_Dt;
        for (j = 0; j < n_Dn - 1; j++)
        {
            stride = strideDrsh[j];
            idiv = idx / stride;
            idx = idx % stride;
            shift += idiv * strideDn[j];
        }
        if (itemsize == 8)
            memcpy(newdataslc + shift + idx, dataslo + idx_Do, 8);
        else
            memcpy(newdataslc + 2 * shift + 2 * idx, dataslo + 2 * idx_Do, 16);
        // itemsize cannot be passed as free parameter -> compiler optimizes transfers for 8-bytes and 16-bytes
        // memcpy(newdataslc + idx_Dt / Drsh1*Dn1 + idx_Dt % Drsh1, dataslo + idx_Do, 8); // avoiding local variables -> the same
        // newdataslc[id0*Dn1 + id1] = dataslo[idx_Do]; // performance - identical as above line
        // int vs int32_t -> no performance change but int32_t limits data to 32 GB

        // next element of Do array  and newdata array
        idx_Do++;
        k = n_order - 1;
        idx_Dt += strideDt[k];
        iDo[k]++;
        while (iDo[k] == Do[k])
        {
            idx_Dt -= Do[k] * strideDt[k];
            iDo[k] = 0;
            k--;
            if (k < 0)
                break;
            iDo[k]++;
            idx_Dt += strideDt[k];
        }
    }
    return 0;
}

DLLEXPORT
int tm_worker_parallel_float64(int tasks, double *newdata, int *sln0, int n_Dn, int *Dn, int *Dslc0,
                               double *data, int *slo0, int *Do, int *order, int *Drsh, int n_order)
{
    int k;
    // tests 2024-09-03
    // 4 threads are optimal (speed/cpu usage) for Xeon cpu
    // 2 threads are optimal (speed/cpu usage) for i5 cpu
#pragma omp parallel for // num_threads(4)
    for (k = 0; k < tasks; k++)
    {
        tm_worker(newdata, sln0[k], n_Dn, Dn + k * n_Dn, Dslc0 + k * n_Dn, data, slo0[k], 
                          Do + k * n_order, order, Drsh + k * n_Dn, n_order, 8);
    }
    return 0;
}

DLLEXPORT
int tm_worker_parallel_complex128(int tasks, double *newdata, int *sln0, int n_Dn, int *Dn, int *Dslc0,
                                  double *data, int *slo0, int *Do, int *order, int *Drsh, int n_order)
{
    int k;

#pragma omp parallel for // num_threads(4)
    // schedule(dynamic,1)
    // schedule(static, 1)   //  static is default and a bit faster
    for (k = 0; k < tasks; k++)
    {
        tm_worker(newdata, sln0[k], n_Dn, Dn + k * n_Dn, Dslc0 + k * n_Dn, data, slo0[k], 
                    Do + k * n_order, order, Drsh + k * n_Dn, n_order, 16);
    }
    return 0;
}

// no longer used
DLLEXPORT
int np_data_2d_cpy(double *dest, double *src, int dest_stride0, int dest_stride1, 
                   int src_stride0, int src_stride1, int src_shape0, int src_shape1)
{
    int i, j;
    char *d;
    char *s;
    int dest_stride = dest_stride0 - dest_stride1 * src_shape1;
    int src_stride = src_stride0 - src_stride1 * src_shape1;

    d = (char *)dest;
    s = (char *)src;
    for (i = src_shape0; i > 0; i--)
    {
        for (j = src_shape1; j > 0; j--)
        {
            memcpy(d, s, 8);
            d += dest_stride1;
            s += src_stride1;
        }
        d += dest_stride;
        s += src_stride;
    }
    return 0;
}

// test function if DLL import and function calling works
DLLEXPORT
void test_empty(void)
{
    puts("Hello from C");
}

#undef DLLEXPORT
#undef MAX_ORDER
