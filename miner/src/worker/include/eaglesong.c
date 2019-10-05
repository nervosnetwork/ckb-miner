#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "eaglesong.h"
#include "portable_endian.h"

#define INPUT_LEN (32)
#define ROUND (43)
#define RATE (256)
#define M (INPUT_LEN >> 2)
#define LEN (RATE >> 3)
#define DELIMITER (0x06)
#define OUTPUT_LENGTH (256 >> 3)
#define N 200000

#define ROL32(a,b) (((a)<<(b))|((a)>>(32-(b))))
#define ROL_ADD(a,b) a += b; a = ROL32(a, 8); b = a + ROL32(b, 24);
#define ROL_XOR(t, a, b, k) t^ROL32(t, a)^ROL32(t, b)^injection_constants[k]

uint32_t injection_constants[] = INJECT_MAT;

#define EaglesongPermutation() { \
    for(int i = 0, k=0; i < ROUND ; ++i ) { \
        tmp = s0^s4^s12^s15; s0 = tmp^s5^s6^s7; s1 = tmp^s1^s8^s13; \
        tmp = s1^s2^s6^s14; s2 = tmp^s7^s8^s9; s3 = tmp^s3^s10^s15; \
        tmp = s0^s3^s4^s8; s4 = tmp^s9^s10^s11; s5 = tmp^s1^s5^s12; \
        tmp = s2^s5^s6^s10; s6 = tmp^s11^s12^s13; s7 = tmp^s3^s7^s14; \
        tmp = s4^s7^s8^s12; s8 = tmp^s13^s14^s15; s9 = tmp^s0^s5^s9; \
        tmp = s6^s9^s10^s14; s10 = tmp^s0^s1^s15; s11 = tmp^s2^s7^s11; \
        tmp = s0^s8^s11^s12; s12 = tmp^s1^s2^s3; s13 = tmp^s4^s9^s13; \
        tmp = s3^s5^s13^s14; s14 = tmp^s2^s4^s10; s15 = tmp^s0^s1^s6^s7^s8^s9^s15; \
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
    ((uint32_t *)output)[k] = htole32(s); \
}

#define absorbing(s, input, i) {\
    s = (be32toh(((uint32_t*)(input))[i])); \
}

uint32_t c_solve(uint8_t *input, uint8_t *target, uint8_t *nonce) {
    uint32_t s0,s1,s2,s3,s4,s5,s6,s7,s8,s9,s10,s11,s12,s13,s14,s15;
    uint32_t state[16];
    uint32_t r0, r1, r2, r3;
    uint32_t tmp;
    uint8_t output[32];
    
    // absorbing
    absorbing(s0, input, 0); absorbing(s1, input, 1);
    absorbing(s2, input, 2); absorbing(s3, input, 3);
    absorbing(s4, input, 4); absorbing(s5, input, 5);
    absorbing(s6, input, 6); absorbing(s7, input, 7);

    s8 = s9 = s10 = s11 = s12 = s13 = s14 = s15 = 0;
    
    EaglesongPermutation();

    RAND_bytes((uint8_t*) &r0, 4);
    RAND_bytes((uint8_t*) &r1, 4);
    RAND_bytes((uint8_t*) &r2, 4);
    RAND_bytes((uint8_t*) &r3, 4);

    s0 ^= r0; s1 ^= r1; s2 ^= r2; s3 ^= r3; s4 ^= DELIMITER;

    state[0] = s0; state[1] = s1; state[2] = s2; state[3] = s3;
    state[4] = s4; state[5] = s5; state[6] = s6; state[7] = s7;
    state[8] = s8; state[9] = s9; state[10] = s10; state[11] = s11;
    state[12] = s12; state[13] = s13; state[14] = s14; state[15] = s15;

    for(uint32_t i=0; i<N; ++i) {
        s0 ^= i;
    
        EaglesongPermutation();

        squeeze(s0, 0); squeeze(s1, 1); squeeze(s2, 2); squeeze(s3, 3);
        squeeze(s4, 4); squeeze(s5, 5); squeeze(s6, 6); squeeze(s7, 7);

        for(int k=0; k<32; ++k) {
            if(output[k] < target[k]) {
                ((uint32_t*)nonce)[0] = le32toh(htobe32((r0^i)));
                ((uint32_t*)nonce)[1] = le32toh(htobe32(r1));
                ((uint32_t*)nonce)[2] = le32toh(htobe32(r2));
                ((uint32_t*)nonce)[3] = le32toh(htobe32(r3));
                return i;
            } else if(output[k] > target[k]) {
                break;
            }
        }

        s0 = state[0]; s1 = state[1]; s2 = state[2]; s3 = state[3];
        s4 = state[4]; s5 = state[5]; s6 = state[6]; s7 = state[7];
        s8 = state[8]; s9 = state[9]; s10 = state[10]; s11 = state[11];
        s12 = state[12]; s13 = state[13]; s14 = state[14]; s15 = state[15];
    }

    return N;
}
