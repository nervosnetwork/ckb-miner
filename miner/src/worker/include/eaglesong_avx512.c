#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "portable_endian.h"
#include "eaglesong.h"

#define INPUT_LEN (32)
#define ROUND (43)
#define RATE (256)
#define M (INPUT_LEN >> 2)
#define LEN (RATE >> 3)
#define DELIMITER (0x06)
#define OUTPUT_LENGTH (256 >> 3)
#define N 3200000

#define ROL32(x, b)   _mm512_rol_epi32 ((x), (b))
#define SL(x, b)      _mm512_slli_epi32 ((x), (b))
#define AND(a, b)     _mm512_and_epi32 ((a), (b))
#define XOR(a, b)     _mm512_xor_epi32 ((a), (b))
#define OR(a, b)      _mm512_or_epi32 ((a), (b))
#define ADD(a, b)     _mm512_add_epi32 ((a), (b))
#define SET1(a)       _mm512_set1_epi32 ((a))
#define STORE         _mm512_store_epi32
#define SET           _mm512_set_epi32
#define u512          __m512i
#define ZERO          _mm512_setzero_epi32()

#define ROL_ADD(a,b) a = ADD(a, b); a = ROL32(a, 8); b = ADD(ROL32(b, 24) ,a);
#define ROL_XOR(t, a, b, k) XOR(XOR(XOR(t, ROL32(t, a)), ROL32(t, b)), SET1(injection_constants_2[k]))

uint32_t injection_constants_2[] = INJECT_MAT;

#define EaglesongPermutation() { \
    for(int i = 0, k=0; i < ROUND ; ++i ) { \
        tmp = XOR(XOR(XOR(s0,s4),s12),s15); s0 = XOR(XOR(XOR(tmp,s5),s6),s7); s1 = XOR(XOR(XOR(tmp,s1),s8),s13); \
        tmp = XOR(XOR(XOR(s1,s2),s6),s14); s2 = XOR(XOR(XOR(tmp,s7),s8),s9); s3 = XOR(XOR(XOR(tmp,s3),s10),s15); \
        tmp = XOR(XOR(XOR(s0,s3),s4),s8); s4 = XOR(XOR(XOR(tmp,s9),s10),s11); s5 = XOR(XOR(XOR(tmp,s1),s5),s12); \
        tmp = XOR(XOR(XOR(s2,s5),s6),s10); s6 = XOR(XOR(XOR(tmp,s11),s12),s13); s7 = XOR(XOR(XOR(tmp,s3),s7),s14); \
        tmp = XOR(XOR(XOR(s4,s7),s8),s12); s8 = XOR(XOR(XOR(tmp,s13),s14),s15); s9 = XOR(XOR(XOR(tmp,s0),s5),s9); \
        tmp = XOR(XOR(XOR(s6,s9),s10),s14); s10 = XOR(XOR(XOR(tmp,s0),s1),s15); s11 = XOR(XOR(XOR(tmp,s2),s7),s11); \
        tmp = XOR(XOR(XOR(s0,s8),s11),s12); s12 = XOR(XOR(XOR(tmp,s1),s2),s3); s13 = XOR(XOR(XOR(tmp,s4),s9),s13); \
        tmp = XOR(XOR(XOR(s3,s5),s13),s14); s14 = XOR(XOR(XOR(tmp,s2),s4),s10); s15 = XOR(XOR(XOR(XOR(XOR(XOR(XOR(tmp,s0),s1),s6),s7),s8),s9),s15); \
        s0 = ROL_XOR(s0, 2, 4, k); ++k; s1 = ROL_XOR(s1, 13, 22, k); ++k; ROL_ADD(s0, s1); \
        s2 = ROL_XOR(s2, 4, 19, k); ++k; s3 = ROL_XOR(s3, 3, 14, k); ++k; ROL_ADD(s2, s3); \
        s4 = ROL_XOR(s4, 27, 31, k); ++k; s5 = ROL_XOR(s5, 3, 8, k); ++k; ROL_ADD(s4, s5); \
        s6 = ROL_XOR(s6, 17, 26, k); ++k; s7 = ROL_XOR(s7, 3, 12, k); ++k; ROL_ADD(s6, s7); \
        s8 = ROL_XOR(s8, 18, 22, k); ++k; s9 = ROL_XOR(s9, 12, 18, k); ++k; ROL_ADD(s8, s9); \
        s10 = ROL_XOR(s10, 4, 7, k); ++k; s11 = ROL_XOR(s11, 4, 31, k); ++k; ROL_ADD(s10, s11); \
        s12 = ROL_XOR(s12, 12, 27, k); ++k; s13 = ROL_XOR(s13, 7, 17, k); ++k; ROL_ADD(s12, s13); \
        s14 = ROL_XOR(s14, 7, 8, k); ++k; s15 = ROL_XOR(s15, 1, 13, k); ++k; ROL_ADD(s14, s15); \
    } \
}

#define squeeze(s, k) {\
    ans = (uint32_t *)&s;\
    for(int i=0; i < 16; ++i) { \
        ((uint32_t *)output[i])[k] = htole32(ans[i]); \
    } \
}

#define absorbing(s, input, i) {\
    s = SET1(be32toh(((uint32_t*)(input))[i])); \
}

uint32_t c_solve_avx512(uint8_t *input, uint8_t *target, uint8_t *nonce) {
    u512 s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;
    u512 state[16];
    uint32_t r0, r1, r2, r3;
    u512 flag, tmp;
    uint8_t output[16][32];
    uint32_t *ans;
    
    // absorbing
    absorbing(s0, input, 0); absorbing(s1, input, 1);
    absorbing(s2, input, 2); absorbing(s3, input, 3);
    absorbing(s4, input, 4); absorbing(s5, input, 5);
    absorbing(s6, input, 6); absorbing(s7, input, 7);

    s8 = s9 = s10 = s11 = s12 = s13 = s14 = s15 = ZERO;
    
    EaglesongPermutation();

    RAND_bytes((uint8_t*) &r0, 4);
    RAND_bytes((uint8_t*) &r1, 4);
    RAND_bytes((uint8_t*) &r2, 4);
    RAND_bytes((uint8_t*) &r3, 4);

    flag = SET(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);

    s0 = XOR(s0, SET1(r0));
    s1 = XOR(s1, SET1(r1));
    s2 = XOR(s2, SET1(r2));
    s3 = XOR(s3, SET1(r3));
    s4 = XOR(s4, SET1(DELIMITER));

    state[0] = s0; state[1] = s1; state[2] = s2; state[3] = s3;
    state[4] = s4; state[5] = s5; state[6] = s6; state[7] = s7;
    state[8] = s8; state[9] = s9; state[10] = s10; state[11] = s11;
    state[12] = s12; state[13] = s13; state[14] = s14; state[15] = s15;

    for(uint32_t i=0; i<N; i+=16) {
        s0 = XOR(s0, ADD(flag, SET1(i)));
        
        EaglesongPermutation();

        squeeze(s0, 0); squeeze(s1, 1); squeeze(s2, 2); squeeze(s3, 3);
        squeeze(s4, 4); squeeze(s5, 5); squeeze(s6, 6); squeeze(s7, 7);

        for(int j=0; j<16; ++j) {
            for(int k=0; k<32; ++k) {
                if(output[j][k] < target[k]) {
                    ((uint32_t*)nonce)[0] = htobe32((r0^(i|j)));
                    ((uint32_t*)nonce)[1] = htobe32(r1);
                    ((uint32_t*)nonce)[2] = htobe32(r2);
                    ((uint32_t*)nonce)[3] = htobe32(r3);
                    return i+16;
                } else if(output[j][k] > target[k]) {
                    break;
                }
            }
        }

        s0 = state[0]; s1 = state[1]; s2 = state[2]; s3 = state[3];
        s4 = state[4]; s5 = state[5]; s6 = state[6]; s7 = state[7];
        s8 = state[8]; s9 = state[9]; s10 = state[10]; s11 = state[11];
        s12 = state[12]; s13 = state[13]; s14 = state[14]; s15 = state[15];
    }

    return N;
}
