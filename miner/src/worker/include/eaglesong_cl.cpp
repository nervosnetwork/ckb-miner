#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <CL/cl.h>
#include "eaglesong.h"
#include "portable_endian.h"

#define INPUT_LEN (32)
#define N (((INPUT_LEN+8+1)+3) >> 2)
#define M (INPUT_LEN >> 2)
#define OUTPUT_LEN (32)
#define THREADS_PER_BLOCK  (256)
#define MAX_GPU_NUM (1024)
#define MAX_PL_NUM (10)
#define HASH_NUM (1<<27)
#define DELIMITER (0x06)


#define KERNEL(...)#__VA_ARGS__
const char *kernelSourceCode = KERNEL(
	__constant unsigned int gpu_injection_constants[] =
{
	0x6e9e40ae, 0x71927c02, 0x9a13d3b1, 0xdaec32ad, 0x3d8951cf, 0xe1c9fe9a, 0xb806b54c, 0xacbbf417,
	0xd3622b3b, 0xa082762a, 0x9edcf1c0, 0xa9bada77, 0x7f91e46c, 0xcb0f6e4f, 0x265d9241, 0xb7bdeab0,
	0x6260c9e6, 0xff50dd2a, 0x9036aa71, 0xce161879, 0xd1307cdf, 0x89e456df, 0xf83133e2, 0x65f55c3d,
	0x94871b01, 0xb5d204cd, 0x583a3264, 0x5e165957, 0x4cbda964, 0x675fca47, 0xf4a3033e, 0x2a417322,
	0x3b61432f, 0x7f5532f2, 0xb609973b, 0x1a795239, 0x31b477c9, 0xd2949d28, 0x78969712, 0x0eb87b6e,
	0x7e11d22d, 0xccee88bd, 0xeed07eb8, 0xe5563a81, 0xe7cb6bcf, 0x25de953e, 0x4d05653a, 0x0b831557,
	0x94b9cd77, 0x13f01579, 0x794b4a4a, 0x67e7c7dc, 0xc456d8d4, 0x59689c9b, 0x668456d7, 0x22d2a2e1,
	0x38b3a828, 0x0315ac3c, 0x438d681e, 0xab7109c5, 0x97ee19a8, 0xde062b2e, 0x2c76c47b, 0x0084456f,
	0x908f0fd3, 0xa646551f, 0x3e826725, 0xd521788e, 0x9f01c2b0, 0x93180cdc, 0x92ea1df8, 0x431a9aae,
	0x7c2ea356, 0xda33ad03, 0x46926893, 0x66bde7d7, 0xb501cc75, 0x1f6e8a41, 0x685250f4, 0x3bb1f318,
	0xaf238c04, 0x974ed2ec, 0x5b159e49, 0xd526f8bf, 0x12085626, 0x3e2432a9, 0x6bd20c48, 0x1f1d59da,
	0x18ab1068, 0x80f83cf8, 0x2c8c11c0, 0x7d548035, 0x0ff675c3, 0xfed160bf, 0x74bbbb24, 0xd98e006b,
	0xdeaa47eb, 0x05f2179e, 0x437b0b71, 0xa7c95f8f, 0x00a99d3b, 0x3fc3c444, 0x72686f8e, 0x00fd01a9,
	0xdedc0787, 0xc6af7626, 0x7012fe76, 0xf2a5f7ce, 0x9a7b2eda, 0x5e57fcf2, 0x4da0d4ad, 0x5c63b155,
	0x34117375, 0xd4134c11, 0x2ea77435, 0x5278b6de, 0xab522c4c, 0xbc8fc702, 0xc94a09e4, 0xebb93a9e,
	0x91ecb65e, 0x4c52ecc6, 0x8703bb52, 0xcb2d60aa, 0x30a0538a, 0x1514f10b, 0x157f6329, 0x3429dc3d,
	0x5db73eb2, 0xa7a1a969, 0x7286bd24, 0x0df6881e, 0x3785ba5f, 0xcd04623a, 0x02758170, 0xd827f556,
	0x99d95191, 0x84457eb1, 0x58a7fb22, 0xd2967c5f, 0x4f0c33f6, 0x4a02099a, 0xe0904821, 0x94124036,
	0x496a031b, 0x780b69c4, 0xcf1a4927, 0x87a119b8, 0xcdfaf4f8, 0x4cf9cd0f, 0x27c96a84, 0x6d11117e,
	0x7f8cf847, 0x74ceede5, 0xc88905e6, 0x60215841, 0x7172875a, 0x736e993a, 0x010aa53c, 0x43d53c2b,
	0xf0d91a93, 0x0d983b56, 0xf816663c, 0xe5d13363, 0x0a61737c, 0x09d51150, 0x83a5ac2f, 0x3e884905,
	0x7b01aeb5, 0x600a6ea7, 0xb7678f7b, 0x72b38977, 0x068018f2, 0xce6ae45b, 0x29188aa8, 0xe5a0b1e9,
	0xc04c2b86, 0x8bd14d75, 0x648781f3, 0xdbae1e0a, 0xddcdd8ae, 0xab4d81a3, 0x446baaba, 0x1cc0c19d,
	0x17be4f90, 0x82c0e65d, 0x676f9c95, 0x5c708db2, 0x6fd4c867, 0xa5106ef0, 0x19dde49d, 0x78182f95,
	0xd089cd81, 0xa32e98fe, 0xbe306c82, 0x6cd83d8c, 0x037f1bde, 0x0b15722d, 0xeddc1e22, 0x93c76559,
	0x8a2f571b, 0x92cc81b4, 0x021b7477, 0x67523904, 0xc95dbccc, 0xac17ee9d, 0x944e46bc, 0x0781867e,
	0xc854dd9d, 0x26e2c30c, 0x858c0416, 0x6d397708, 0xebe29c58, 0xc80ced86, 0xd496b4ab, 0xbe45e6f5,
	0x10d24706, 0xacf8187a, 0x96f523cb, 0x2227e143, 0x78c36564, 0x4643adc2, 0x4729d97a, 0xcff93e0d,
	0x25484bbd, 0x91c6798e, 0x95f773f4, 0x44204675, 0x2eda57ba, 0x06d313ef, 0xeeaa4466, 0x2dfa7530,
	0xa8af0c9b, 0x39f1535e, 0x0cc2b7bd, 0x38a76c0e, 0x4f41071d, 0xcdaf2475, 0x49a6eff8, 0x01621748,
	0x36ebacab, 0xbd6d9a29, 0x44d1cd65, 0x40815dfd, 0x55fa5a1a, 0x87cce9e9, 0xae559b45, 0xd76b4c26,
	0x637d60ad, 0xde29f5f9, 0x97491cbb, 0xfb350040, 0xffe7f997, 0x201c9dcd, 0xe61320e9, 0xa90987a3,
	0xe24afa83, 0x61c1e6fc, 0xcc87ff62, 0xf1c9d8fa, 0x4fd04546, 0x90ecc76e, 0x46e456b9, 0x305dceb8,
	0xf627e68c, 0x2d286815, 0xc705bbfd, 0x101b6df3, 0x892dae62, 0xd5b7fb44, 0xea1d5c94, 0x5332e3cb,
	0xf856f88a, 0xb341b0e9, 0x28408d9d, 0x5421bc17, 0xeb9af9bc, 0x602371c5, 0x67985a91, 0xd774907f,
	0x7c4d697d, 0x9370b0b8, 0x6ff5cebb, 0x7d465744, 0x674ceac0, 0xea9102fc, 0x0de94784, 0xc793de69,
	0xfe599bb1, 0xc6ad952f, 0x6d6ca9c3, 0x928c3f91, 0xf9022f05, 0x24a164dc, 0xe5e98cd3, 0x7649efdb,
	0x6df3bcdb, 0x5d1e9ff1, 0x17f5d010, 0xe2686ea1, 0x6eac77fe, 0x7bb5c585, 0x88d90cbb, 0x18689163,
	0x67c9efa5, 0xc0b76d9b, 0x960efbab, 0xbd872807, 0x70f4c474, 0x56c29d20, 0xd1541d15, 0x88137033,
	0xe3f02b3e, 0xb6d9b28d, 0x53a077ba, 0xeedcd29e, 0xa50a6c1d, 0x12c2801e, 0x52ba335b, 0x35984614,
	0xe2599aa8, 0xaf94ed1d, 0xd90d4767, 0x202c7d07, 0x77bec4f4, 0xfa71bc80, 0xfc5c8b76, 0x8d0fbbfc,
	0xda366dc6, 0x8b32a0c7, 0x1b36f7fc, 0x6642dcbc, 0x6fe7e724, 0x8b5fa782, 0xc4227404, 0x3a7d1da7,
	0x517ed658, 0x8a18df6d, 0x3e5c9b23, 0x1fbd51ef, 0x1470601d, 0x3400389c, 0x676b065d, 0x8864ad80,
	0xea6f1a9c, 0x2db484e1, 0x608785f0, 0x8dd384af, 0x69d26699, 0x409c4e16, 0x77f9986a, 0x7f491266,
	0x883ea6cf, 0xeaa06072, 0xfa2e5db5, 0x352594b4, 0x9156bb89, 0xa2fbbbfb, 0xac3989c7, 0x6e2422b1,
	0x581f3560, 0x1009a9b5, 0x7e5ad9cd, 0xa9fc0a6e, 0x43e5998e, 0x7f8778f9, 0xf038f8e1, 0x5415c2e8,
	0x6499b731, 0xb82389ae, 0x05d4d819, 0x0f06440e, 0xf1735aa0, 0x986430ee, 0x47ec952c, 0xbf149cc5,
	0xb3cb2cb6, 0x3f41e8c2, 0x271ac51b, 0x48ac5ded, 0xf76a0469, 0x717bba4d, 0x4f5c90d6, 0x3b74f756,
	0x1824110a, 0xa4fd43e3, 0x1eb0507c, 0xa9375c08, 0x157c59a7, 0x0cad8f51, 0xd66031a0, 0xabb5343f,
	0xe533fa43, 0x1996e2bb, 0xd7953a71, 0xd2529b94, 0x58f0fa07, 0x4c9b1877, 0x057e990d, 0x8bfe19c4,
	0xa8e2c0c9, 0x99fcaada, 0x69d2aaca, 0xdc1c4642, 0xf4d22307, 0x7fe27e8c, 0x1366aa07, 0x1594e637,
	0xce1066bf, 0xdb922552, 0x9930b52a, 0xaeaa9a3e, 0x31ff7eb4, 0x5e1f945a, 0x150ac49c, 0x0ccdac2d,
	0xd8a8a217, 0xb82ea6e5, 0xd6a74659, 0x67b7e3e6, 0x836eef4a, 0xb6f90074, 0x7fa3ea4b, 0xcb038123,
	0xbf069f55, 0x1fa83fc4, 0xd6ebdb23, 0x16f0a137, 0x19a7110d, 0x5ff3b55f, 0xfb633868, 0xb466f845,
	0xbce0c198, 0x88404296, 0xddbdd88b, 0x7fc52546, 0x63a553f8, 0xa728405a, 0x378a2bce, 0x6862e570,
	0xefb77e7d, 0xc611625e, 0x32515c15, 0x6984b765, 0xe8405976, 0x9ba386fd, 0xd4eed4d9, 0xf8fe0309,
	0x0ce54601, 0xbaf879c2, 0xd8524057, 0x1d8c1d7a, 0x72c0a3a9, 0x5a1ffbde, 0x82f33a45, 0x5143f446,
	0x29c7e182, 0xe536c32f, 0x5a6f245b, 0x44272adb, 0xcb701d9c, 0xf76137ec, 0x0841f145, 0xe7042ecc,
	0xf1277dd7, 0x745cf92c, 0xa8fe65fe, 0xd3e2d7cf, 0x54c513ef, 0x6079bc2d, 0xb66336b0, 0x101e383b,
	0xbcd75753, 0x25be238a, 0x56a6f0be, 0xeeffcc17, 0x5ea31f3d, 0x0ae772f5, 0xf76de3de, 0x1bbecdad,
	0xc9107d43, 0xf7e38dce, 0x618358cd, 0x5c833f04, 0xf6975906, 0xde4177e5, 0x67d314dc, 0xb4760f3e,
	0x56ce5888, 0x0e8345a8, 0xbff6b1bf, 0x78dfb112, 0xf1709c1e, 0x7bb8ed8b, 0x902402b9, 0xdaa64ae0,
	0x46b71d89, 0x7eee035f, 0xbe376509, 0x99648f3a, 0x0863ea1f, 0x49ad8887, 0x79bdecc5, 0x3c10b568,
	0x5f2e4bae, 0x04ef20ab, 0x72f8ce7b, 0x521e1ebe, 0x14525535, 0x2e8af95b, 0x9094ccfd, 0xbcf36713,
	0xc73953ef, 0xd4b91474, 0x6554ec2d, 0xe3885c96, 0x03dc73b7, 0x931688a9, 0xcbbef182, 0x2b77cfc9,
	0x632a32bd, 0xd2115dcc, 0x1ae5533d, 0x32684e13, 0x4cc5a004, 0x13321bde, 0x62cbd38d, 0x78383a3b,
	0xd00686f1, 0x9f601ee7, 0x7eaf23de, 0x3110c492, 0x9c351209, 0x7eb89d52, 0x6d566eac, 0xc2efd226,
	0x32e9fac5, 0x52227274, 0x09f84725, 0xb8d0b605, 0x72291f02, 0x71b5c34b, 0x3dbfcbb8, 0x04a02263,
	0x55ba597f, 0xd4e4037d, 0xc813e1be, 0xffddeefa, 0xc3c058f3, 0x87010f2e, 0x1dfcf55f, 0xc694eeeb,
	0xa9c01a74, 0x98c2fc6b, 0xe57e1428, 0xdd265a71, 0x836b956d, 0x7e46ab1a, 0x5835d541, 0x50b32505,
	0xe640913c, 0xbb486079, 0xfe496263, 0x113c5b69, 0x93cd6620, 0x5efe823b, 0x2d657b40, 0xb46dfc6c,
	0x57710c69, 0xfe9fadeb, 0xb5f8728a, 0xe3224170, 0xca28b751, 0xfdabae56, 0x5ab12c3c, 0xa697c457,
	0xd28fa2b7, 0x056579f2, 0x9fd9d810, 0xe3557478, 0xd88d89ab, 0xa72a9422, 0x6d47abd0, 0x405bcbd9,
	0x6f83ebaf, 0x13caec76, 0xfceb9ee2, 0x2e922df7, 0xce9856df, 0xc05e9322, 0x2772c854, 0xb67f2a32,
	0x6d1af28d, 0x3a78cf77, 0xdff411e4, 0x61c74ca9, 0xed8b842e, 0x72880845, 0x6e857085, 0xc6404932,
	0xee37f6bc, 0x27116f48, 0x5e9ec45a, 0x8ea2a51f, 0xa5573db7, 0xa746d036, 0x486b4768, 0x5b438f3b,
	0x18c54a5c, 0x64fcf08e, 0xe993cdc1, 0x35c1ead3, 0x9de07de7, 0x321b841c, 0x87423c5e, 0x071aa0f6,
	0x962eb75b, 0xbb06bdd2, 0xdcdb5363, 0x389752f2, 0x83d9cc88, 0xd014adc6, 0xc71121bb, 0x2372f938,
	0xcaff2650, 0x62be8951, 0x56dccaff, 0xac4084c0, 0x09712e95, 0x1d3c288f, 0x1b085744, 0xe1d3cfef,
	0x5c9a812e, 0x6611fd59, 0x85e46044, 0x1981d885, 0x5a4c903f, 0x43f30d4b, 0x7d1d601b, 0xdd3c3391,
	0x030ec65e, 0xc12878cd, 0x72e795fe, 0xd0c76abd, 0x1ec085db, 0x7cbb61fa, 0x93e8dd1e, 0x8582eb06,
	0x73563144, 0x049d4e7e, 0x5fd5aefe, 0x7b842a00, 0x75ced665, 0xbb32d458, 0x4e83bba7, 0x8f15151f,
	0x7795a125, 0xf0842455, 0x499af99d, 0x565cc7fa, 0xa3b1278d, 0x3f27ce74, 0x96ca058e, 0x8a497443,
	0xa6fb8cae, 0xc115aa21, 0x17504923, 0xe4932402, 0xaea886c2, 0x8eb79af5, 0xebd5ea6b, 0xc7980d3b,
	0x71369315, 0x796e6a66, 0x3a7ec708, 0xb05175c8, 0xe02b74e7, 0xeb377ad3, 0x6c8c1f54, 0xb980c374,
	0x59aee281, 0x449cb799, 0xe01f5605, 0xed0e085e, 0xc9a1a3b4, 0xaac481b1, 0xc935c39c, 0xb7d8ce7f
};
__kernel void eaglesongcl(__global uint *state, __global unsigned char* target, __global uint *nonce_id)
{

	uint global_id = get_global_id(0);
	uint id = get_local_id(0);
	uint tmp;
	uint s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14, s15;
	unsigned char output[32]; //OUTPUT_LEN
	int i, j, k;

	__local uint shared_state[11];//N
	__local unsigned char shared_target[32];//OUTPUT_LEN

	if (id < 11) shared_state[id] = state[id];//N
	if (id < 32) shared_target[id] = target[id];
	barrier(CLK_LOCAL_MEM_FENCE);

	
	s0 = shared_state[0] ^ (global_id + 1);
	s1 = shared_state[1]; s2 = shared_state[2]; s3 = shared_state[3];
	s4 = shared_state[4]; s5 = shared_state[5]; s6 = shared_state[6]; s7 = shared_state[7];
	s8 = s9 = s10 = s11 = s12 = s13 = s14 = s15 = 0;
	

	k = 0;
	while (k < 2)
	{
		for (i = 0, j = 0; i < 43; ++i, j += 16)//ROUND
		{
			tmp = s0 ^ s4 ^ s12 ^ s15; s0 = tmp^s5 ^ s6 ^ s7; s1 = tmp^s1 ^ s8 ^ s13;
			tmp = s1 ^ s2 ^ s6 ^ s14; s2 = tmp^s7 ^ s8 ^ s9; s3 = tmp^s3 ^ s10 ^ s15;
			tmp = s0 ^ s3 ^ s4 ^ s8; s4 = tmp^s9 ^ s10 ^ s11; s5 = tmp^s1 ^ s5 ^ s12;
			tmp = s2 ^ s5 ^ s6 ^ s10; s6 = tmp^s11 ^ s12 ^ s13; s7 = tmp^s3 ^ s7 ^ s14;
			tmp = s4 ^ s7 ^ s8 ^ s12; s8 = tmp^s13 ^ s14 ^ s15; s9 = tmp^s0 ^ s5 ^ s9;
			tmp = s6 ^ s9 ^ s10 ^ s14; s10 = tmp^s0 ^ s1 ^ s15; s11 = tmp^s2 ^ s7 ^ s11;
			tmp = s0 ^ s8 ^ s11 ^ s12; s12 = tmp^s1 ^ s2 ^ s3; s13 = tmp^s4 ^ s9 ^ s13;
			tmp = s3 ^ s5 ^ s13 ^ s14; s14 = tmp^s2 ^ s4 ^ s10; s15 = tmp^s0 ^ s1 ^ s6 ^ s7 ^ s8 ^ s9 ^ s15;

			s0 ^= (((s0) << (2)) | ((s0) >> (32 - (2)))) ^ (((s0) << (4)) | ((s0) >> (32 - (4)))) ^ gpu_injection_constants[(j ^ 0)];
			s1 ^= (((s1) << (13)) | ((s1) >> (32 - (13)))) ^ (((s1) << (22)) | ((s1) >> (32 - (22)))) ^ gpu_injection_constants[(j ^ 1)];
			s0 += s1; s0 = (((s0) << (8)) | ((s0) >> (32 - (8)))); s1 = (((s1) << (24)) | ((s1) >> (32 - (24)))) + s0;

			s2 ^= (((s2) << (4)) | ((s2) >> (32 - (4)))) ^ (((s2) << (19)) | ((s2) >> (32 - (19)))) ^ gpu_injection_constants[(j ^ 2)];
			s3 ^= (((s3) << (3)) | ((s3) >> (32 - (3)))) ^ (((s3) << (14)) | ((s3) >> (32 - (14)))) ^ gpu_injection_constants[(j ^ 3)];
			s2 += s3; s2 = (((s2) << (8)) | ((s2) >> (32 - (8)))); s3 = (((s3) << (24)) | ((s3) >> (32 - (24)))) + s2;

			s4 ^= (((s4) << (27)) | ((s4) >> (32 - (27)))) ^ (((s4) << (31)) | ((s4) >> (32 - (31)))) ^ gpu_injection_constants[(j ^ 4)];
			s5 ^= (((s5) << (3)) | ((s5) >> (32 - (3)))) ^ (((s5) << (8)) | ((s5) >> (32 - (8)))) ^ gpu_injection_constants[(j ^ 5)];
			s4 += s5; s4 = (((s4) << (8)) | ((s4) >> (32 - (8)))); s5 = (((s5) << (24)) | ((s5) >> (32 - (24)))) + s4;

			s6 ^= (((s6) << (17)) | ((s6) >> (32 - (17)))) ^ (((s6) << (26)) | ((s6) >> (32 - (26)))) ^ gpu_injection_constants[(j ^ 6)];
			s7 ^= (((s7) << (3)) | ((s7) >> (32 - (3)))) ^ (((s7) << (12)) | ((s7) >> (32 - (12)))) ^ gpu_injection_constants[(j ^ 7)];
			s6 += s7; s6 = (((s6) << (8)) | ((s6) >> (32 - (8)))); s7 = (((s7) << (24)) | ((s7) >> (32 - (24)))) + s6;



			s8 ^= (((s8) << (18)) | ((s8) >> (32 - (18)))) ^ (((s8) << (22)) | ((s8) >> (32 - (22)))) ^ gpu_injection_constants[(j ^ 8)];
			s9 ^= (((s9) << (12)) | ((s9) >> (32 - (12)))) ^ (((s9) << (18)) | ((s9) >> (32 - (18)))) ^ gpu_injection_constants[(j ^ 9)];
			s8 += s9; s8 = (((s8) << (8)) | ((s8) >> (32 - (8)))); s9 = (((s9) << (24)) | ((s9) >> (32 - (24)))) + s8;

			s10 ^= (((s10) << (4)) | ((s10) >> (32 - (4)))) ^ (((s10) << (7)) | ((s10) >> (32 - (7)))) ^ gpu_injection_constants[(j ^ 10)];
			s11 ^= (((s11) << (4)) | ((s11) >> (32 - (4)))) ^ (((s11) << (31)) | ((s11) >> (32 - (31)))) ^ gpu_injection_constants[(j ^ 11)];
			s10 += s11; s10 = (((s10) << (8)) | ((s10) >> (32 - (8)))); s11 = (((s11) << (24)) | ((s11) >> (32 - (24)))) + s10;

			s12 ^= (((s12) << (12)) | ((s12) >> (32 - (12)))) ^ (((s12) << (27)) | ((s12) >> (32 - (27)))) ^ gpu_injection_constants[(j ^ 12)];
			s13 ^= (((s13) << (7)) | ((s13) >> (32 - (7)))) ^ (((s13) << (17)) | ((s13) >> (32 - (17)))) ^ gpu_injection_constants[(j ^ 13)];
			s12 += s13; s12 = (((s12) << (8)) | ((s12) >> (32 - (8)))); s13 = (((s13) << (24)) | ((s13) >> (32 - (24)))) + s12;

			s14 ^= (((s14) << (7)) | ((s14) >> (32 - (7)))) ^ (((s14) << (8)) | ((s14) >> (32 - (8)))) ^ gpu_injection_constants[(j ^ 14)];
			s15 ^= (((s15) << (1)) | ((s15) >> (32 - (1)))) ^ (((s15) << (13)) | ((s15) >> (32 - (13)))) ^ gpu_injection_constants[(j ^ 15)];
			s14 += s15; s14 = (((s14) << (8)) | ((s14) >> (32 - (8)))); s15 = (((s15) << (24)) | ((s15) >> (32 - (24)))) + s14;
		}
		if (k==0)
		{ 
			s0 ^= shared_state[8]; s1 ^= shared_state[9]; s2 ^= shared_state[10]; 
		}
		k++;
	}


	((uint *)output)[0] = (s0); ((uint *)output)[1] = (s1); ((uint *)output)[2] = (s2); ((uint *)output)[3] = (s3);
	((uint *)output)[4] = (s4); ((uint *)output)[5] = (s5); ((uint *)output)[6] = (s6); ((uint *)output)[7] = (s7);

	for (k = 0; k<32; ++k)
	{
		if (output[k] < shared_target[k]) 
		{
			atomic_xchg(nonce_id, global_id + 1);
			break;
		}
		else if (output[k] > shared_target[k]) 
		{
			break;
		}
	}
}
);

struct OCL_PLAT
{
	cl_uint          numPlatforms = 0;           // the number of platforms
	cl_int           status = clGetPlatformIDs(0, NULL, &numPlatforms);
	cl_device_id     *devices;
	cl_platform_id   *platforms;
	cl_uint           num_devices, addr_data;
	size_t            namelen, deviceListSize;
	char              pform_name[256];
	char              dev_name[256];
	cl_context        context;
	cl_program        program = NULL;
};
OCL_PLAT *plat_info[MAX_PL_NUM] = {0};

struct GPU_DEVICE
{
	uint32_t           state[N];
	uint32_t           nonce_id;
	uint8_t            *target;
	cl_kernel          kernel = NULL;
	cl_mem             g_nonce_id = NULL;
	cl_mem             g_target = NULL;
	cl_mem             g_state = NULL;
	cl_command_queue   commandQueue = NULL;
};
GPU_DEVICE *gpu_divices[MAX_PL_NUM][MAX_GPU_NUM] = {0};

GPU_DEVICE* New_GPU_DEVICE()
{
	GPU_DEVICE* p = NULL;
	p = (GPU_DEVICE*)malloc(sizeof(GPU_DEVICE));
	if (p != NULL)
	{
		p->kernel = NULL;
		p->g_nonce_id = NULL;
		p->g_target = NULL;
		p->g_state = NULL;
		p->commandQueue = NULL;
	}
	else 
	{
		printf("E08: alloc memory error!\n");
	}
	return p;
}

uint32_t plat_device(cl_program program, cl_context context, cl_device_id *devices, size_t gpu_id, size_t p_id, cl_int  status, uint8_t *input, uint8_t *target, uint64_t *nonce)
{
	size_t globalThreads[] = { HASH_NUM };
	size_t localThreads[] = { THREADS_PER_BLOCK };
	GPU_DEVICE *dev;

	if (gpu_divices[p_id][gpu_id] == NULL) gpu_divices[p_id][gpu_id] = New_GPU_DEVICE();
	dev = gpu_divices[p_id][gpu_id];

	if (dev->kernel == NULL)dev->kernel = clCreateKernel(program, "eaglesongcl", &status);
	if (status != CL_SUCCESS) { printf("Error: Creating Kernel from program.(clCreateKernel)\n"); goto end; }

	if (dev->commandQueue == NULL)dev->commandQueue = clCreateCommandQueue(context, devices[gpu_id], 0, &status);
	if (status != CL_SUCCESS) { printf("Error: Create Command Queue. (clCreateCommandQueue)\n"); goto end; }
	
	if (dev->g_nonce_id == NULL)dev->g_nonce_id = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, sizeof(uint32_t), NULL, &status);
	if (status != CL_SUCCESS) { printf("Error: Create Buffer, g_nonce_id. (clCreateBuffer)\n"); goto end; }

	if (dev->g_target == NULL)dev->g_target = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, OUTPUT_LEN, NULL, &status);
	if (status != CL_SUCCESS) { printf("Error: Create Buffer, g_target. (clCreateBuffer)\n"); goto end; }

	if (dev->g_state == NULL)dev->g_state = clCreateBuffer(context, CL_MEM_ALLOC_HOST_PTR, N * sizeof(uint32_t), NULL, &status);
	if (status != CL_SUCCESS) { printf("Error: Create Buffer, g_state. (clCreateBuffer)\n"); goto end; }

	RAND_bytes((uint8_t*)&(dev->state[0]), 4);
	RAND_bytes((uint8_t*)&(dev->state[1]), 4);

	for (int j = 0, k = 0; j <= M; ++j) {
		uint32_t sum = 0;
		for (int v = 0; v < 4; ++v) {
			if (k < INPUT_LEN) {
				sum = (sum << 8) ^ input[k];
			}
			else if (k == INPUT_LEN) {
				sum = (sum << 8) ^ DELIMITER;
			}
			++k;
		}
		dev->state[j + 2] = sum;
	}

	dev->target = target;
	dev->nonce_id = 0;

	status = clEnqueueWriteBuffer(dev->commandQueue, dev->g_nonce_id, CL_TRUE, 0, sizeof(uint32_t), &dev->nonce_id, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("Error: clEnqueueWriteBuffer nonce_id. (clSetKernelArg)\n"); goto end; }

	status = clEnqueueWriteBuffer(dev->commandQueue, dev->g_target, CL_TRUE, 0, OUTPUT_LEN, (dev->target), 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("Error: clEnqueueWriteBuffer g_target. (clSetKernelArg)\n"); goto end; }

	status = clEnqueueWriteBuffer(dev->commandQueue, dev->g_state, CL_TRUE, 0, N * sizeof(uint32_t), &dev->state, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("Error: clEnqueueWriteBuffer nonce_id. (clSetKernelArg)\n"); goto end; }

	status = clSetKernelArg(dev->kernel, 0, sizeof(cl_mem), (void *)&dev->g_state);
	if (status != CL_SUCCESS) { printf("Error: Setting kernel argument g_state. (clSetKernelArg)\n"); goto end; }

	status = clSetKernelArg(dev->kernel, 1, sizeof(cl_mem), (void *)&dev->g_target);
	if (status != CL_SUCCESS) { printf("Error: Setting kernel argument g_target. (clSetKernelArg)\n"); goto end; }

	status = clSetKernelArg(dev->kernel, 2, sizeof(cl_mem), (void *)&dev->g_nonce_id);
	if (status != CL_SUCCESS) { printf("Error: Setting kernel argument g_nonce_id. (clSetKernelArg)\n"); goto end; }
	
	status = clEnqueueNDRangeKernel(dev->commandQueue, dev->kernel, 1, NULL, globalThreads, localThreads, 0, NULL, NULL);

	if (status != CL_SUCCESS) { printf("Error: Enqueueing kernel\n"); goto end; }
	status = clFinish(dev->commandQueue);
	if (status != CL_SUCCESS) { printf("Error: Finish command queue\n"); goto end; }

	status = clEnqueueReadBuffer(dev->commandQueue, dev->g_nonce_id, CL_TRUE, 0, sizeof(dev->nonce_id), &dev->nonce_id, 0, NULL, NULL);
	if (status != CL_SUCCESS) { printf("Error: Read buffer queue\n"); goto end; }

	if (dev->nonce_id)
	{
		*nonce = le32toh(htobe32(dev->state[1]));
		*nonce = (*nonce << 32) ^ le32toh(htobe32(((dev->state[0]) ^ (dev->nonce_id))));
	}

	return HASH_NUM;
	
end:
	clReleaseKernel(dev->kernel);
	clReleaseMemObject(dev->g_state);
	clReleaseMemObject(dev->g_target);
	clReleaseMemObject(dev->g_nonce_id);
	clReleaseCommandQueue(dev->commandQueue);
	free(gpu_divices[p_id][gpu_id]);
	gpu_divices[p_id][gpu_id] = NULL;
	return 0;
}

extern "C" {
	void c_opencl_initial()
	{
		memset(gpu_divices, 0, sizeof(gpu_divices));
		memset(plat_info, 0, sizeof(plat_info));
	}

	uint32_t c_plat_init(uint32_t plat_id) {
		if(plat_id > MAX_PL_NUM) {
			printf("plat_id is too large!\n");
			return 0;
		}

		if (plat_info[plat_id] == NULL)
		{
			plat_info[plat_id] = (OCL_PLAT*)malloc(sizeof(OCL_PLAT));
			memset(plat_info[plat_id], 0, sizeof(OCL_PLAT));

			plat_info[plat_id]->numPlatforms = 0;
			plat_info[plat_id]->status = clGetPlatformIDs(0, NULL, &plat_info[plat_id]->numPlatforms);
			if (plat_info[plat_id]->status != CL_SUCCESS)printf("clGetPlatformIDs error = %5d\n", plat_info[plat_id]->status);
			printf("Number of platforms: %5d\n", plat_info[plat_id]->numPlatforms);

			plat_info[plat_id]->platforms = (cl_platform_id*)malloc(plat_info[plat_id]->numPlatforms * sizeof(cl_platform_id));
			plat_info[plat_id]->status = clGetPlatformIDs(plat_info[plat_id]->numPlatforms, plat_info[plat_id]->platforms, NULL);
			if (plat_info[plat_id]->status!= CL_SUCCESS){ printf("Couldn't find any platforms\n"); goto error; }

			if (plat_id < plat_info[plat_id]->numPlatforms) {
				plat_info[plat_id]->status = clGetPlatformInfo(plat_info[plat_id]->platforms[plat_id], CL_PLATFORM_NAME, sizeof(plat_info[plat_id]->pform_name), &plat_info[plat_id]->pform_name, NULL);
				if (plat_info[plat_id]->status != CL_SUCCESS){ printf("CL_PLATFORM_NAME error code %5d\n", plat_info[plat_id]->status); goto error; }
				

				plat_info[plat_id]->status = clGetDeviceIDs(plat_info[plat_id]->platforms[plat_id], CL_DEVICE_TYPE_GPU, 0, NULL, &plat_info[plat_id]->num_devices);
				if (plat_info[plat_id]->status != CL_SUCCESS){ printf("Couldn't find any num_devices error code %5d\n", plat_info[plat_id]->status); goto error; }
				printf("Platform: %s, Number of devices: %d\n", plat_info[plat_id]->pform_name, plat_info[plat_id]->num_devices);
				printf(" \n");
				
				cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(plat_info[plat_id]->platforms[plat_id]), 0 };
				plat_info[plat_id]->context = clCreateContextFromType(prop, CL_DEVICE_TYPE_ALL, NULL, NULL, &plat_info[plat_id]->status);
				plat_info[plat_id]->status = clGetContextInfo(plat_info[plat_id]->context, CL_CONTEXT_DEVICES, 0, NULL, &plat_info[plat_id]->deviceListSize);
				if (plat_info[plat_id]->status != CL_SUCCESS) { printf("Error: Getting Context Info device list size, clGetContextInfo)\n"); goto error; }

				plat_info[plat_id]->devices = (cl_device_id *)malloc(plat_info[plat_id]->deviceListSize);
				if (plat_info[plat_id]->devices == 0) { printf("Error: No devices found.\n"); goto error; }

				plat_info[plat_id]->status = clGetContextInfo(plat_info[plat_id]->context, CL_CONTEXT_DEVICES, plat_info[plat_id]->deviceListSize, plat_info[plat_id]->devices, NULL);

				if (plat_info[plat_id]->status != CL_SUCCESS) { printf("Error: Getting Context Info (device list, clGetContextInfo)\n"); goto error; }

				size_t sourceSize[] = { strlen(kernelSourceCode) };
				plat_info[plat_id]->program = clCreateProgramWithSource(plat_info[plat_id]->context, 1, &kernelSourceCode, sourceSize, &plat_info[plat_id]->status);

				if (plat_info[plat_id]->status != CL_SUCCESS) { printf("Error: Loading Binary into cl_program (clCreateProgramWithBinary)\n"); goto error; }
				plat_info[plat_id]->status = clBuildProgram(plat_info[plat_id]->program, plat_info[plat_id]->num_devices, plat_info[plat_id]->devices, NULL, NULL, NULL);
				if (plat_info[plat_id]->status != CL_SUCCESS) { printf("Error: Building Program (clBuildingProgram)\n"); goto error; }
			} else {
				printf("plat_id is too large, max: %d\n", plat_info[plat_id]->numPlatforms);
				goto error;
			}
		}

		return 1;

	error:
		clReleaseProgram(plat_info[plat_id]->program);
		clReleaseContext(plat_info[plat_id]->context);
		free(plat_info[plat_id]->devices);
		free(plat_info[plat_id]->platforms);
		plat_info[plat_id] = NULL;
		return 0;
	}

	uint32_t c_solve_cl(uint8_t *input, uint8_t *target, uint64_t *nonce, uint32_t plat_id, uint32_t gpuid)
	{
		if (plat_info[plat_id] == NULL)
		{
			printf("Platform is not init!\n");
			return 0;
		}

		if (gpuid < plat_info[plat_id]->num_devices) {
			return plat_device(plat_info[plat_id]->program, plat_info[plat_id]->context, plat_info[plat_id]->devices, gpuid, plat_id, plat_info[plat_id]->status,input, target, nonce);
		} else { 
			printf("Error: device %d\n not found", gpuid);
			return 0;
		}

	}
}
