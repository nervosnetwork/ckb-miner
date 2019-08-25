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
#include "portable_endian.h"
#include "eaglesong.h"

#define INPUT_LEN (32)
#define N ((INPUT_LEN+8+1)+3) >> 2
#define M (INPUT_LEN >> 2)
#define OUTPUT_LEN 32
#define THREADS_PER_BLOCK  (512)
#define MAX_HASH_NUM (1<<30)
#define MAX_GPU_NUM (1024)
#define HASH_NUM (1<<27)

#define DELIMITER (0x06)

#define ROUND (43)


#define ROL32(a,b) (((a)<<(b))|((a)>>(32-(b))))
#define ROL_ADD(a,b) a += b; a = ROL32(a, 8); b = ROL32(b, 24) + a;

#define EaglesongPermutation() \
{ \
	for(int i = 0, j=0; i < ROUND ; ++i, j+=16) { \
		tmp = s0 ^ s4 ^ s12 ^ s15; s0 = tmp^s5 ^ s6 ^ s7; s1 = tmp^s1 ^ s8 ^ s13; \
		tmp = s1 ^ s2 ^ s6 ^ s14; s2 = tmp^s7 ^ s8 ^ s9; s3 = tmp^s3 ^ s10 ^ s15; \
		tmp = s0 ^ s3 ^ s4 ^ s8; s4 = tmp^s9 ^ s10 ^ s11; s5 = tmp^s1 ^ s5 ^ s12; \
		tmp = s2 ^ s5 ^ s6 ^ s10; s6 = tmp^s11 ^ s12 ^ s13; s7 = tmp^s3 ^ s7 ^ s14; \
		tmp = s4 ^ s7 ^ s8 ^ s12; s8 = tmp^s13 ^ s14 ^ s15; s9 = tmp^s0 ^ s5 ^ s9; \
		tmp = s6 ^ s9 ^ s10 ^ s14; s10 = tmp^s0 ^ s1 ^ s15; s11 = tmp^s2 ^ s7 ^ s11; \
		tmp = s0 ^ s8 ^ s11 ^ s12; s12 = tmp^s1 ^ s2 ^ s3; s13 = tmp^s4 ^ s9 ^ s13; \
		tmp = s3 ^ s5 ^ s13 ^ s14; s14 = tmp^s2 ^ s4 ^ s10; s15 = tmp^s0 ^ s1 ^ s6 ^ s7 ^ s8 ^ s9 ^ s15; \
		s0 ^= ROL32(s0, 2) ^ ROL32(s0, 4) ^ gpu_injection_constants[(j ^ 0)];                    \
		s1 ^= ROL32(s1, 13) ^ ROL32(s1, 22) ^ gpu_injection_constants[(j ^ 1)];                  \
		ROL_ADD(s0, s1);                                                                      \
		s2 ^= ROL32(s2, 4) ^ ROL32(s2, 19) ^ gpu_injection_constants[(j ^ 2)];                   \
		s3 ^= ROL32(s3, 3) ^ ROL32(s3, 14) ^ gpu_injection_constants[(j ^ 3)];                   \
		ROL_ADD(s2, s3);                                                                      \
		s4 ^= ROL32(s4, 27) ^ ROL32(s4, 31) ^ gpu_injection_constants[(j ^ 4)];                  \
		s5 ^= ROL32(s5, 3) ^ ROL32(s5, 8) ^ gpu_injection_constants[(j ^ 5)];                    \
		ROL_ADD(s4, s5);                                                                      \
		s6 ^= ROL32(s6, 17) ^ ROL32(s6, 26) ^ gpu_injection_constants[(j ^ 6)];                  \
		s7 ^= ROL32(s7, 3) ^ ROL32(s7, 12) ^ gpu_injection_constants[(j ^ 7)];                   \
		ROL_ADD(s6, s7);                                                                      \
		s8 ^= ROL32(s8, 18) ^ ROL32(s8, 22) ^ gpu_injection_constants[(j ^ 8)];                  \
		s9 ^= ROL32(s9, 12) ^ ROL32(s9, 18) ^ gpu_injection_constants[(j ^ 9)];                  \
		ROL_ADD(s8, s9);                                                                      \
		s10 ^= ROL32(s10, 4) ^ ROL32(s10, 7) ^ gpu_injection_constants[(j ^ 10)];                 \
		s11 ^= ROL32(s11, 4) ^ ROL32(s11, 31) ^ gpu_injection_constants[(j ^ 11)];                \
		ROL_ADD(s10, s11);                                                                    \
		s12 ^= ROL32(s12, 12) ^ ROL32(s12, 27) ^ gpu_injection_constants[(j ^ 12)];               \
		s13 ^= ROL32(s13, 7) ^ ROL32(s13, 17) ^ gpu_injection_constants[(j ^ 13)];                \
		ROL_ADD(s12, s13);                                                                    \
		s14 ^= ROL32(s14, 7) ^ ROL32(s14, 8) ^ gpu_injection_constants[(j ^ 14)];                 \
		s15 ^= ROL32(s15, 1) ^ ROL32(s15, 13) ^ gpu_injection_constants[(j ^ 15)];                \
		ROL_ADD(s14, s15); \
	} \
}

__constant__ uint32_t gpu_injection_constants[688] = INJECT_MAT;

#define squeeze(s, k) {\
    ((uint32_t *)output)[k] = (s); \
}


struct GPU_DEVICE
{
	uint32_t   state[N];
	uint32_t   nonce_id;
	uint8_t    *target;
	uint32_t    *g_state;
	uint8_t    *g_target;
	uint32_t   *g_nonce_id;
};

GPU_DEVICE *gpu_divices[MAX_GPU_NUM] = {NULL};
uint32_t gpu_divices_cnt = 0;


__global__ void g_eaglesong(uint32_t *state, uint8_t* target, uint32_t *nonce_id)
{
	uint32_t global_id = blockDim.x * blockIdx.x + threadIdx.x;

	uint32_t id = global_id % THREADS_PER_BLOCK;
	uint32_t tmp;
	uint32_t s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
	uint8_t output[OUTPUT_LEN];

	__shared__ uint32_t shared_state[N];
	__shared__ uint8_t shared_target[OUTPUT_LEN];

	if (id < N) shared_state[id] = state[id]; 
	if (id < OUTPUT_LEN) shared_target[id] = target[id];
	__syncthreads();

	s0 = shared_state[0] ^ (global_id+1);
	s1 = shared_state[1]; s2 = shared_state[2]; s3 = shared_state[3];
	s4 = shared_state[4]; s5 = shared_state[5]; s6 = shared_state[6]; s7 = shared_state[7];
	s8 = s9 = s10 = s11 = s12 = s13 = s14 = s15 = 0;
	
	EaglesongPermutation();
	
	s0 ^= shared_state[8]; s1 ^= shared_state[9]; s2 ^= shared_state[10];
	
	EaglesongPermutation();

	squeeze(s0, 0); squeeze(s1, 1); squeeze(s2, 2); squeeze(s3, 3);
	squeeze(s4, 4); squeeze(s5, 5); squeeze(s6, 6); squeeze(s7, 7);

	for(int k=0; k<32; ++k) {
		if(output[k] < shared_target[k]) {
			atomicExch(nonce_id, global_id+1);
		} else if(output[k] > shared_target[k]) {
			break;
		}
	}
}

int gpu_hash(uint32_t gpuid)
{
	if (HASH_NUM > MAX_HASH_NUM) {
		printf("HASH_NUM out of bound!!!\n");
		return 0;
	}

	if (gpu_divices[gpuid]->g_state == NULL)
	{
		if (cudaMalloc((void **)&gpu_divices[gpuid]->g_state, sizeof(gpu_divices[gpuid]->state)) != cudaSuccess) {
			printf("E01: cuda alloc memory error for state\n");
			return 0;
		}
	}

	if (gpu_divices[gpuid]->g_nonce_id == NULL) 
	{
		if (cudaMalloc((void **)&gpu_divices[gpuid]->g_nonce_id, sizeof(gpu_divices[gpuid]->nonce_id)) != cudaSuccess) {
			printf("E02: cuda alloc memory error for nonce\n");
			return 0;
		}
	}

	if (gpu_divices[gpuid]->g_target == NULL)
	{
		if (cudaMalloc((void **)&gpu_divices[gpuid]->g_target, OUTPUT_LEN) != cudaSuccess) {
			printf("E03: cuda alloc memory error for target\n");
			return 0;
		}
	}

	if (cudaMemcpy(gpu_divices[gpuid]->g_state, gpu_divices[gpuid]->state, sizeof(gpu_divices[gpuid]->state), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("E04: copy memory error for state\n");
		return 0;
	}

	if (cudaMemcpy(gpu_divices[gpuid]->g_target, gpu_divices[gpuid]->target, OUTPUT_LEN, cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("E05: copy memory error for target\n");
		return 0;
	}

	if (cudaMemcpy(gpu_divices[gpuid]->g_nonce_id, &(gpu_divices[gpuid]->nonce_id), sizeof(gpu_divices[gpuid]->nonce_id), cudaMemcpyHostToDevice) != cudaSuccess)
	{
		printf("E06: copy memory error for nonce\n");
		return 0;
	}

	g_eaglesong << <HASH_NUM / THREADS_PER_BLOCK, THREADS_PER_BLOCK >> >(gpu_divices[gpuid]->g_state, gpu_divices[gpuid]->g_target, gpu_divices[gpuid]->g_nonce_id);
	cudaDeviceSynchronize();

	if (cudaMemcpy(&(gpu_divices[gpuid]->nonce_id), gpu_divices[gpuid]->g_nonce_id, sizeof(gpu_divices[gpuid]->nonce_id), cudaMemcpyDeviceToHost) != cudaSuccess)
	{
		printf("E07: copy memory error for g_nonce_id\n");
		return 0;
	}

	return HASH_NUM;
}

GPU_DEVICE* New_GPU_DEVICE()
{
	GPU_DEVICE* p = NULL;
	p = (GPU_DEVICE*)malloc(sizeof(GPU_DEVICE));
	if (p != NULL)
	{
		p->g_target = NULL;
		p->g_nonce_id = NULL;
		p->g_state = NULL;
		p->g_target = NULL;
	} else {
		printf("E08: alloc memory error!\n");
	}
	return p;
}

void RESET_GPU_DEVICE(uint32_t gpuid)
{
	
	memset(gpu_divices[gpuid]->state, 0, sizeof(gpu_divices[gpuid]->state));
	gpu_divices[gpuid]->nonce_id = 0;

	cudaFree(gpu_divices[gpuid]->g_nonce_id);
	cudaFree(gpu_divices[gpuid]->g_state);
	cudaFree(gpu_divices[gpuid]->g_target);
	
	gpu_divices[gpuid]->target = NULL;
	gpu_divices[gpuid]->g_nonce_id = NULL;
	gpu_divices[gpuid]->g_state = NULL;
	gpu_divices[gpuid]->g_target = NULL;
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
		printf("name:%s\n", prop.name);
		printf("totalGlobalMem:%lu GB\n", prop.totalGlobalMem / 1024 / 1024 / 1024);
		printf("multiProcessorCount:%d\n", prop.multiProcessorCount);
		printf("maxThreadsPerBlock:%d\n", prop.maxThreadsPerBlock);
		printf("sharedMemPerBlock:%lu KB\n", prop.sharedMemPerBlock/1024);
		printf("major:%d,minor:%d\n", prop.major, prop.minor);
		gpu_divices_cnt++;
	}
	if (gpu_divices_cnt > MAX_GPU_NUM)gpu_divices_cnt = MAX_GPU_NUM;
}

extern "C" {
	uint32_t c_solve_gpu(uint8_t *input, uint8_t *target, uint64_t *nonce, uint32_t gpuid) {
		while(!gpu_divices[gpuid]) {
			gpu_divices[gpuid] = New_GPU_DEVICE();
		}

		uint32_t ret;
		RAND_bytes((uint8_t*) &(gpu_divices[gpuid]->state[0]), 4);
		RAND_bytes((uint8_t*) &(gpu_divices[gpuid]->state[1]), 4);
		
		// absorbing
		for(int j = 0, k=0; j <= M; ++j) {
			uint32_t sum = 0;
			for(int v=0; v < 4; ++v) {
				if(k < INPUT_LEN) {
					sum = (sum << 8) ^ input[k];
				} else if(k == INPUT_LEN) {
					sum = (sum << 8) ^ DELIMITER;
				}
				++k;
			}
			gpu_divices[gpuid]->state[j+2] = sum;
		}
		gpu_divices[gpuid]->target = target;
		gpu_divices[gpuid]->nonce_id = 0;

		ret = gpu_hash(gpuid);

		if(gpu_divices[gpuid]->nonce_id) {
			*nonce = le32toh(htobe32(gpu_divices[gpuid]->state[1]));
			*nonce = (*nonce << 32) ^ le32toh(htobe32(((gpu_divices[gpuid]->state[0])^(gpu_divices[gpuid]->nonce_id))));
		}

		return ret;
	}
}