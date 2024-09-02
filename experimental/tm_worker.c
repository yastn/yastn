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

DLLEXPORT
void test_empty(void)
{
    puts("Hello from C");
}


// this is a slower version that is no longer used
DLLEXPORT
int tm_worker_alpha(double *newdata, int sln0, int Dn1, int Dslc0, int Dslc1,
                 double *data, int slo0, int *Do, int *order, int Drsh1, int n_order)
{
    int iDo[MAX_ORDER]; 
    int end = 0;
    int i, j, k, idx_Do, idx_Dt, id0, id1;

    if (n_order > MAX_ORDER)
    {
        printf("tm_worker error: Too big n_order.");
        return -1;
    }
    double *dataslo = data + slo0;
    double *newdataslc = newdata + sln0 + Dslc0 * Dn1 + Dslc1;

    // first element of Do
    idx_Do = 0;
    for (i = 0; i < n_order; i++)
    {
        iDo[i] = 0;
    }
    // loop over Do elements
    while (!end)
    {
        idx_Dt = 0;
        for (j = 0; j < n_order; j++)
        {
            idx_Dt = idx_Dt * Do[order[j]] + iDo[order[j]];
        }
        id0 = idx_Dt / Drsh1;
        id1 = idx_Dt % Drsh1;
        memcpy(newdataslc + id0 * Dn1 + id1, dataslo + idx_Do, 8); 
        // newdataslc[id0*Dn1 + id1] = dataslo[idx_Do]; // identical asm code

        // next element of Do array
        idx_Do++;
        k = n_order - 1;
        iDo[k] += 1;
        while ((k >= 0) && (iDo[k] == Do[k]))
        {
            iDo[k] = 0;
            k--;
            if (k < 0)
            {
                end = 1;
            }
            else
            {
                iDo[k] = iDo[k] + 1;
            }
        }
    }
    return 0;
}

// temp = newdata[slice(*sln)].reshape(Dn)
// temp[slcs] = data[slice(*slo)].reshape(Do).transpose(order).reshape(Drsh)
DLLEXPORT
int tm_worker(double *newdata, int sln0, int Dn1, int Dslc0, int Dslc1,
                double *data, int slo0, int *Do, int *order, int Drsh1, int n_order, int itemsize)
{
    int iDo[MAX_ORDER];       // indexes of Do array are used to loop
    int strandsDt[MAX_ORDER]; // strands of Do.transpose(order) array
    int j, k, idx_Do, idx_Dt, id0, id1, nDt;

    if (n_order > MAX_ORDER)
    {
        printf("tm_worker error: n_order > MAX_ORDER=32. Results will be WRONG.\n");
        return -1;
    }
    if ((itemsize != 8) && (itemsize != 16))
    {
        printf("tm_worker error: itemsize != 8 && itemsize != 16.\n");
        return -1;
    }

    double *dataslo;
    double *newdataslc;
    if (itemsize == 8)
    {
        dataslo = data + slo0;
        newdataslc = newdata + sln0 + Dslc0 * Dn1 + Dslc1;
    }
    else
    { // itemsize==16
        dataslo = data + 2 * slo0;
        newdataslc = newdata + 2 * (sln0 + Dslc0 * Dn1 + Dslc1);
    }

    memset(iDo, 0, sizeof(int) * MAX_ORDER);
    memset(strandsDt, 0, sizeof(int) * MAX_ORDER);
    nDt = 1;
    for (j = 0; j < n_order; j++)
    {
        k = order[n_order - j - 1];
        strandsDt[k] = nDt;
        nDt *= Do[k];
    }
    idx_Do = 0;
    idx_Dt = 0;
    id0 = 0;
    id1 = 0;

    if (itemsize == 8)
    {
        // loop over elements
        k = 1; // dummy value for start
        while (k >= 0)
        {
            /*idx_Dt = 0;
            for (j=0; j<n_order; j++) {
                idx_Dt = idx_Dt*Do[order[j]] + iDo[order[j]];
            }  //   older and slower algorithm */
            // separate Drsh1==4 case to avoid IDIV -> no improvement

            id0 = idx_Dt / Drsh1;
            id1 = idx_Dt % Drsh1;
            memcpy(newdataslc + id0 * Dn1 + id1, dataslo + idx_Do, 8);
            //  itemsize cannot be passed as free parameter -> compiler optimizes transfers for 8-bytes and 16-bytes differently
            // memcpy(newdataslc + idx_Dt / Drsh1*Dn1 + idx_Dt % Drsh1, dataslo + idx_Do, 8); // avoiding local variables -> the same
            // newdataslc[id0*Dn1 + id1] = dataslo[idx_Do]; // performance - identical as above line
            // int vs int -> no performance change

            // next element of Do array  and newdata array
            idx_Do++;
            k = n_order - 1;
            idx_Dt += strandsDt[k];
            iDo[k]++;
            while (iDo[k] == Do[k])
            {
                idx_Dt -= Do[k] * strandsDt[k];
                iDo[k] = 0;
                k--;
                if (k < 0)
                    break;
                iDo[k]++;
                idx_Dt += strandsDt[k];
            }
        }
    }
    else
    { // itemsize==16
        k = 1;
        while (k >= 0)
        {
            id0 = idx_Dt / Drsh1;
            id1 = idx_Dt % Drsh1;
            memcpy(newdataslc + 2 * id0 * Dn1 + 2 * id1, dataslo + 2 * idx_Do, 16);
            idx_Do++;
            k = n_order - 1;
            idx_Dt += strandsDt[k];
            iDo[k]++;
            while (iDo[k] == Do[k])
            {
                idx_Dt -= Do[k] * strandsDt[k];
                iDo[k] = 0;
                k--;
                if (k < 0)
                    break;
                iDo[k]++;
                idx_Dt += strandsDt[k];
            }
        }
    }
    return 0;
}

DLLEXPORT
int tm_worker_parallel_float64(int tasks, double *newdata, int *sln0, int *Dn1, int *Dslc0, int *Dslc1,
                                    double *data, int *slo0, int *Do, int *order, int *Drsh1, int n_order)
{
    int k;

    #pragma omp parallel for 
	// num_threads(2)
    for (k = 0; k < tasks; k++)
    {
        tm_worker(newdata, sln0[k], Dn1[k], Dslc0[k], Dslc1[k], data, slo0[k], Do + k * n_order, order, Drsh1[k], n_order, 8);
    }
    return 0;
}

DLLEXPORT
int tm_worker_parallel_complex128(int tasks, double *newdata, int *sln0, int *Dn1, int *Dslc0, int *Dslc1,
                                       double *data, int *slo0, int *Do, int *order, int *Drsh1, int n_order)
{
    int k;

    #pragma omp parallel for 
	// num_threads(2)
    // schedule(dynamic,1)
    // schedule(static, 1)   // test 2024-09-01   static is a bit faster
    for (k = 0; k < tasks; k++)
    {
        tm_worker(newdata, sln0[k], Dn1[k], Dslc0[k], Dslc1[k], data, slo0[k], Do + k * n_order, order, Drsh1[k], n_order, 16);
    }
    return 0;
}

// no longer used
DLLEXPORT
int np_data_2d_cpy(double *dest, double *src, int dest_stride0, int dest_stride1, int src_stride0, int src_stride1, int src_shape0, int src_shape1)
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

#undef DLLEXPORT
#undef MAX_ORDER

/* 2024-09-01 benchmark  complex128, 1000 iterations, a_Dsize=954145  (281 tasks)
pure numpy(openblas): 9.65 s
c worker_parallel: 1 thread: 6.65 s, already faster because  xmm registers are used for 16-byte data transfer
c worker_parallel: 2 threads: 5.36 s
c worker_parallel: 3 threads: 4.94 s
c worker_parallel: 4 threads: 4.88 s
*/
