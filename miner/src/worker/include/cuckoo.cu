/// System includes
#include <stdio.h>
#include <assert.h>

// CUDA runtime
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stddef.h>
#include <openssl/rand.h>
#include "blake2b.h"

#define threadsPerBlock  (512)
#define MaxCuckooNum (4*4096)
#define MaxGpuNum (1024)
#define trim (32)
#define SolveThreadsPerBlock (128)
#define SolveEN (128)
#define CuckooNum (2*4096)

#define rotl(x, b) (((x) << (b)) | ((x) >> (64 - (b))))
#define EBIT 15
#define CLEN 12
#define EN (1 << EBIT)
#define M (EN << 1)
#define MASK ((1 << EBIT) - 1)
#define CN CLEN << 2

struct GPU_DEVICE
{
    uint32_t cproof[CuckooNum][CLEN];
    uint8_t msg[CuckooNum][32];
    uint8_t alive[CuckooNum][EN >> 3];
    uint8_t calive[CuckooNum][EN >> 3];
    uint64_t nonces[CuckooNum];

    uint8_t  *gmsg = NULL;
    uint8_t  *gRHash = NULL;
    uint32_t *gRege = NULL;
    uint32_t *gproof = NULL;
    uint32_t *gnode = NULL;
};

GPU_DEVICE *gpu_divices[MaxGpuNum] = {NULL};
uint32_t gpu_divices_cnt = 0;

// set siphash keys from 32 byte char array
#define setkeys() \
    k0 = (((uint64_t *)mesg)[0]); \
    k1 = (((uint64_t *)mesg)[1]); \
    k2 = (((uint64_t *)mesg)[2]); \
    k3 = (((uint64_t *)mesg)[3]);

#define sip_round() \
    v0 += v1; v2 += v3; v1 = rotl(v1, 13); \
    v3 = rotl(v3, 16); v1 ^= v0; v3 ^= v2; \
    v0 = rotl(v0, 32); v2 += v1; v0 += v3; \
    v1 = rotl(v1, 17); v3 = rotl(v3, 21); \
    v1 ^= v2; v3 ^= v0; v2 = rotl(v2, 32);

#define siphash24( nonce , hashv ) { \
    v0 = k0; v1 = k1; v2 = k2; v3 = k3; \
    v3 ^= (nonce); \
    sip_round(); sip_round(); \
    v0 ^= (nonce); \
    v2 ^= 0xff; \
    sip_round(); sip_round(); sip_round(); sip_round(); \
    hashv = (v0 ^ v1 ^ v2  ^ v3); \
}

__global__ void kill_leaf(uint8_t *gmesg, uint32_t *gRege, uint8_t *gRHash, uint32_t *gnode)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    uint64_t k0, k1, k2, k3;
    uint64_t v0, v1, v2, v3;
    uint64_t hash;
    uint64_t st, ed;
    uint32_t tmp, tw_bit, py, py2;
    uint32_t i, j;
    uint32_t u;
    uint32_t uorv, edgeidx;

    uint32_t block_tid = id % threadsPerBlock;
    uint32_t block_id = id / threadsPerBlock;
    uint32_t block_ENrange = EN / threadsPerBlock;
    uint32_t block_AliveRange = (EN >> 3) / threadsPerBlock;
    uint32_t block_AliveNode = (EN >> 4) / threadsPerBlock;
    uint8_t *mesg = gmesg + block_id * 32;

    uint32_t *node32 = gnode + block_id * M;

    __shared__ uint8_t alive_ege[EN >> 3];
    __shared__ uint32_t node[EN >> 4];
    __shared__ uint32_t RHash[SolveEN][2];
    __shared__ uint32_t RegeSP;

    if (block_tid == 0)atomicAnd(&RegeSP, 0);

    setkeys();
    st = block_tid*block_ENrange; ed = (block_tid + 1)*block_ENrange;

    memset(alive_ege + block_tid*block_AliveRange, 0, block_AliveRange);
    __syncthreads();
    for (i = st; i < ed; i++)
    {
        siphash24((i << 1) + 0, hash);
        u = (hash & MASK);
        node32[i << 1] = u;
        siphash24((i << 1) + 1, hash);
        u = (hash & MASK);
        node32[(i << 1) + 1] = u;
    }
    __syncthreads();

    for (j = 0; j < trim; j++)
    {
        uorv = 0;
        memset(node + block_tid*block_AliveNode, 0, block_AliveNode*sizeof(uint32_t)); __syncthreads();

        for (i = st; i < ed; i++)
        {
            if (!((alive_ege[i >> 3] >> (i & 7)) & 1))
            {
                u = node32[(i << 1) + uorv];
                py = 1 << ((u << 1) & 31);
                tmp = atomicOr(&node[u >> 4], py);
                py2 = py << 1;
                if ((tmp & (py2 | py)) == py)atomicOr(&node[u >> 4], py2);
            }
        }
        __syncthreads();
        for (i = st; i < ed; i++)
        {
            if (!((alive_ege[i >> 3] >> (i & 7)) & 1))
            {
                u = node32[(i << 1) + uorv];
                tmp = node[u >> 4];
                py = ((u << 1) & 31);
                tw_bit = (tmp >> py) & 2;
                if (!tw_bit)
                {
                    alive_ege[i >> 3] = alive_ege[i >> 3] ^ (1 << (i & 7));
                }
            }
        }
        __syncthreads();
        uorv = 1;
        memset(node + block_tid*block_AliveNode, 0, block_AliveNode*sizeof(uint32_t)); __syncthreads();

        for (i = st; i < ed; i++)
        {
            if (!((alive_ege[i >> 3] >> (i & 7)) & 1))
            {
                u = node32[(i << 1) + uorv];
                py = 1 << ((u << 1) & 31);
                tmp = atomicOr(&node[u >> 4], py);
                py2 = py << 1;
                if ((tmp & (py2 | py)) == py)atomicOr(&node[u >> 4], py2);
            }
        }
        __syncthreads();
        for (i = st; i < ed; i++)
        {
            if (!((alive_ege[i >> 3] >> (i & 7)) & 1))
            {
                u = node32[(i << 1) + uorv];
                tmp = node[u >> 4];
                py = ((u << 1) & 31);
                tw_bit = (tmp >> py) & 2;
                if (!tw_bit)
                {
                    alive_ege[i >> 3] = alive_ege[i >> 3] ^ (1 << (i & 7));
                }
            }
        }
        __syncthreads();
    }
    __syncthreads();

    gRege[SolveEN*block_id + block_tid % SolveEN] = 0xffffffff;
    __syncthreads();
    for (i = st; i < ed; i++)
    {
        if (!((alive_ege[i >> 3] >> (i & 7)) & 1))
        {
            edgeidx = atomicInc(&RegeSP, 0xffffffff)%126;
            gRege[SolveEN * block_id + edgeidx] = i;
            u = node32[(i << 1) + 0] << 1;
            RHash[edgeidx][0] = u;
            u = node32[(i << 1) + 1] << 1 + 1;
            RHash[edgeidx][1] = u;
        }
    }
    __syncthreads();

    if (block_tid <= 1)
    {
        tmp = 0;
        if (RegeSP > 126)edgeidx = 126;
        else    edgeidx = RegeSP;
        for (i = 0; i < edgeidx; i++)
        {
            py = RHash[i][block_tid];
            if (py == 0xffffffff)continue;
            RHash[i][block_tid] = 0xffffffff;
            gRHash[(SolveEN << 1) *block_id + (i << 1) + block_tid] = (tmp << 1) + block_tid;
            for (j = i + 1; j < edgeidx; j++)
            {
                py2 = RHash[j][block_tid];
                if (py2 == py)
                {
                    RHash[j][block_tid] = 0xffffffff;
                    gRHash[(SolveEN << 1) *block_id + (j << 1) + block_tid] = (tmp << 1) + block_tid;
                }
            }
            tmp++;
        }
    }
    __syncthreads();

}

__global__ void solve128X_127EN(uint32_t *gRege, uint8_t *gRHash, uint32_t *gproof)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t i, tmp;
    uint8_t u, v;

    uint32_t block_tid = id % SolveThreadsPerBlock;
    uint32_t *Rege = gRege + id * SolveEN;
    uint8_t *RHash = gRHash + id * (SolveEN << 1);
    uint32_t *proof = gproof + id * CLEN;

    __shared__ uint32_t path[SolveThreadsPerBlock][CLEN];
    __shared__ uint8_t graph[SolveThreadsPerBlock][SolveEN << 1];

    uint8_t pre;
    uint8_t cur;
    uint8_t next;

    memset(&graph[block_tid], 0xff, (SolveEN << 1));
    proof[0] = 0xffffffff;

    for (i = 0; i<SolveEN; i++)
    {
        if (Rege[i] == 0xffffffff)
        {
            break;
        }
        u = RHash[i << 1];
        v = RHash[(i << 1) + 1];
        __syncthreads();
        pre = 0xff;
        cur = u;
        while (cur != 0xff)
        {
            next = graph[block_tid][cur];
            graph[block_tid][cur] = pre;
            pre = cur;
            cur = next;
        }
        int m = 0;
        cur = v;
        while (graph[block_tid][cur] != 0xff && m < CLEN)
        {
            cur = graph[block_tid][cur];
            ++m;
        }
        if (cur != u)
        {
            graph[block_tid][u] = v;
        }
        else if (m == CLEN - 1)
        {
            int j;
            cur = v;
            for (j = 0; j <= m; ++j)
            {
                path[block_tid][j] = cur;
                cur = graph[block_tid][cur];
            }

            memset(&graph[block_tid], 0xff, (SolveEN << 1));

            for (j = 1; j <= m; ++j)
            {
                graph[block_tid][path[block_tid][j]] = path[block_tid][j - 1];
            }

            int k = 0;
            int b = CLEN - 1;
            for (j = 0; k < b; ++j)
            {
                u = RHash[j << 1];
                v = RHash[(j << 1) + 1];
                if (graph[block_tid][u] == v)
                {
                    path[block_tid][k] = Rege[j];
                    graph[block_tid][u] = 0xff;
                    ++k;
                }
                else if(graph[block_tid][v] == u)
                {
                    path[block_tid][k] = Rege[j];
                    graph[block_tid][v] = 0xff;
                    ++k;
                }
            }
            path[block_tid][k] = Rege[i];

            for (j = 0; j < CLEN - 1; j++) // sort
            {
                for (k = 0; k < CLEN - j - 1; k++)
                {
                    if (path[block_tid][k]>path[block_tid][k + 1])
                    {
                        tmp = path[block_tid][k];
                        path[block_tid][k] = path[block_tid][k + 1];
                        path[block_tid][k + 1] = tmp;
                    }
                }
            }
            for (j = 0; j < CLEN; j++)proof[j] = path[block_tid][j];
            break;
        }
    }
    __syncthreads();
}

int gpu_cuckoo(uint32_t gpuid)
{
    if (CuckooNum > MaxCuckooNum) {
        printf("CuckooNum out of bound!!!\n");
        return 0;
    }

    if (CuckooNum % SolveThreadsPerBlock != 0) {
        printf("CuckooNum must be a multiple of SolveThreadsPerBlock = %5d\n", SolveThreadsPerBlock);
        return 0;
    }

    //alloc once
    if (gpu_divices[gpuid]->gmsg == NULL) {
        if (cudaMalloc((void **)&gpu_divices[gpuid]->gmsg, CuckooNum * 32 * sizeof(uint8_t)) != cudaSuccess) {
            printf("gmsg cudaMalloc error\n");
            return 0;
        }
    }

    if (gpu_divices[gpuid]->gRege == NULL) {
        if (cudaMalloc((void **)&gpu_divices[gpuid]->gRege, CuckooNum * 128 * sizeof(uint32_t)) != cudaSuccess) {
            printf("gRege cudaMalloc error\n");
            return 0;
        }
    }

    if (gpu_divices[gpuid]->gRHash == NULL) {
        if (cudaMalloc((void **)&gpu_divices[gpuid]->gRHash, CuckooNum * 2 * 128 * sizeof(uint8_t)) != cudaSuccess) {
            printf("gRHash cudaMalloc error\n");
            return 0;
        }
    }

    if (gpu_divices[gpuid]->gproof == NULL) {
        if (cudaMalloc((void **)&gpu_divices[gpuid]->gproof, CuckooNum * CLEN * sizeof(uint32_t)) != cudaSuccess) {
            printf("gproof cudaMalloc error\n");
            return 0;
        }
    }

    if (gpu_divices[gpuid]->gnode == NULL) {
        if (cudaMalloc((void **)&gpu_divices[gpuid]->gnode, CuckooNum * M * sizeof(uint32_t)) != cudaSuccess) {
            printf("gnode cudaMalloc error\n");
            return 0;
        }
    }

    if (cudaMemcpy(gpu_divices[gpuid]->gmsg, gpu_divices[gpuid]->msg, CuckooNum * 32 * sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        printf("copy memory error\n");
        return 0;
    }

    kill_leaf << <CuckooNum, threadsPerBlock >> >(gpu_divices[gpuid]->gmsg, gpu_divices[gpuid]->gRege, gpu_divices[gpuid]->gRHash, gpu_divices[gpuid]->gnode);
    cudaDeviceSynchronize();

    solve128X_127EN << <CuckooNum / SolveThreadsPerBlock, SolveThreadsPerBlock >> >(gpu_divices[gpuid]->gRege, gpu_divices[gpuid]->gRHash, gpu_divices[gpuid]->gproof);
    cudaDeviceSynchronize();

    if (cudaMemcpy(gpu_divices[gpuid]->cproof, gpu_divices[gpuid]->gproof, CuckooNum * CLEN * sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        printf("copy memory error\n");
        return 0;
    }
    return CuckooNum;
}

GPU_DEVICE* New_GPU_DEVICE()
{
    GPU_DEVICE* p = NULL;
    p = (GPU_DEVICE*)malloc(sizeof(GPU_DEVICE));
    if (p != NULL)
    {
        p->gmsg = NULL;
        p->gRHash = NULL;
        p->gRege = NULL;
        p->gproof = NULL;
        p->gnode = NULL;
    }
    return p;
}
void GPU_Count()
{
    int num;
    cudaDeviceProp prop;
    cudaGetDeviceCount(&num);
    printf("deviceCount := %d\n", num);
    gpu_divices_cnt = 0;
    for (int i = 0; i<num; i++)
    {

        cudaGetDeviceProperties(&prop, i);
        printf("name: %s\n", prop.name);
        printf("totalGlobalMem: %luG\n", prop.totalGlobalMem / 1024 / 1024 / 1024);
        printf("multiProcessorCount: %d\n", prop.multiProcessorCount);
        printf("maxThreadsPerBlock: %d\n", prop.maxThreadsPerBlock);
        printf("major:%d,minor: %d\n", prop.major, prop.minor);
        gpu_divices_cnt++;
    }
}

extern "C" {
    int c_solve_gpu(uint32_t *prof, uint64_t *nonc, const uint8_t *hash, const uint8_t *target, uint32_t gpuid) {
        while (!gpu_divices[gpuid]) {
            gpu_divices[gpuid] = New_GPU_DEVICE();
        }

        uint8_t pmesg[CN];
        uint8_t thash[32];
        blake2b_state S;
        
        b2b_setup(&S);
        memcpy(pmesg+8, hash, 32);

        for(int i=0; i< CuckooNum; ++i) {
            RAND_bytes(pmesg, 8);
            blake2b_state tmp = S;
            b2b_update(&tmp, pmesg, 40);
            b2b_final(&tmp, gpu_divices[gpuid]->msg[i], 32);
            gpu_divices[gpuid]->nonces[i] = le64toh(((uint64_t *)pmesg)[0]);
        }
        
        cudaSetDevice(gpuid);
        int ret = gpu_cuckoo(gpuid);

        for(int i=0; i< ret; ++i) {
            if (gpu_divices[gpuid]->cproof[i][0] != 0xffffffff)
            {
                memcpy(pmesg, gpu_divices[gpuid]->cproof[i], CN);
                blake2b_state tmp = S;
                b2b_update(&tmp, pmesg, CN);
                b2b_final(&tmp, thash, 32);

                for(int j=0; j<32; ++j) {
                    if(thash[j] < target[j]) {
                        memcpy(prof, gpu_divices[gpuid]->cproof[i], CN);
                        *nonc = gpu_divices[gpuid]->nonces[i];
                        prof[CLEN] = 1;
                        return ret;
                    } else if(thash[j] > target[j]) {
                        break;
                    }
                }
            }
        }

        return ret;
    }
}
