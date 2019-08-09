#include <stdint.h>    // for types uint32_t,uint64_t
#include "portable_endian.h"    // for htole32/64

#include <stdlib.h>
#include <string.h>
#include <openssl/rand.h>
#include "blake2b.h"

#define EBIT 15
#define CLEN 12

#define EN 1 << EBIT
#define CN CLEN << 2
#define M EN << 1
#define MASK ((1 << EBIT) - 1)

#define rotl(x, b) ((x) << (b)) | ((x) >> (64 - (b)))

// set siphash keys from 32 byte char array
#define setkeys() \
  k0 = le64toh(((uint64_t *)mesg)[0]); \
  k1 = le64toh(((uint64_t *)mesg)[1]); \
  k2 = le64toh(((uint64_t *)mesg)[2]); \
  k3 = le64toh(((uint64_t *)mesg)[3]);


#define sip_round() \
  v0 += v1; v2 += v3; v1 = rotl(v1,13); \
  v3 = rotl(v3,16); v1 ^= v0; v3 ^= v2; \
  v0 = rotl(v0,32); v2 += v1; v0 += v3; \
  v1 = rotl(v1,17); v3 = rotl(v3,21); \
  v1 ^= v2; v3 ^= v0; v2 = rotl(v2,32); 

#define siphash24( nonce ) ({\
  v0 = k0; v1 = k1; v2 = k2; v3 = k3; \
  v3 ^= (nonce); \
  sip_round(); sip_round(); \
  v0 ^= (nonce); \
  v2 ^= 0xff; \
  sip_round(); sip_round(); sip_round(); sip_round(); \
  (v0 ^ v1 ^ v2  ^ v3); \
})

int c_solve(uint32_t *prof, uint64_t *nonc, const uint8_t *hash, const uint8_t *target) {
  int graph[M];
  int V[EN], U[EN];
  int path[CLEN];

  uint8_t pmesg[40];
  uint8_t mesg[32];
  uint8_t hmesg[CN];

  blake2b_state S;

  uint64_t k0, k1, k2, k3;
  uint64_t v0, v1, v2, v3;

  b2b_setup(&S);
  
  memcpy(pmesg+8, hash, 32);
  
  for(int gs=1; gs<200; ++gs) {
    RAND_bytes(pmesg, 8);
    blake2b_state tmp = S;
    b2b_update(&tmp, pmesg, 40);
    b2b_final(&tmp, mesg, 32);
    setkeys();
    
    for(int i=0; i<M; ++i) {
        graph[i] = -1;
    }
    
    for(uint64_t i=0; i<EN; ++i) {
        U[i] = ( siphash24((i << 1)) & MASK) << 1;
        V[i] = (((siphash24(((i<<1)+1))) & MASK) << 1) + 1;
    }
    
    for(uint64_t i=0; i<EN; ++i) {
        int u = U[i];
        int v = V[i];

        int pre = -1;
        int cur = u;
        int next;
        while(cur != -1) {
            next = graph[cur];
            graph[cur] = pre;
            pre = cur;
            cur = next;
        }

        int m = 0;
        cur = v;
        while(graph[cur] != -1 && m < CLEN) {
            cur = graph[cur];
            ++m;
        }

        if(cur != u) {
            graph[u] = v;
        } else if(m == CLEN-1) {
            int j;
            
            cur = v;
            for(j=0; j<=m; ++j) {
                path[j] = cur;
                cur = graph[cur];
            }

            for(j=0; j<M; ++j) {
                graph[j] = -1;
            }
            
            for(j=1; j<=m; ++j) {
                graph[path[j]] = path[j-1];
            }

            int k = 0;
            int b = CLEN -1;
            for(j=0; k < b; ++j) {
                int u = U[j];
                int v = V[j];

                if (graph[u] == v) {
                    prof[k] = j;
                    graph[u] = -1;
                    ++k;
                } else if (graph[v] == u) {
                    prof[k] = j;
                    graph[v] = -1;
                    ++k;
                }
                
            }
            prof[k] = i;
            
            memcpy(hmesg, prof, CN);
            blake2b_state tmp = S;
            b2b_update(&tmp, hmesg, CN);
            b2b_final(&tmp, mesg, 32);

            for(int k=0; k<32; ++k) {
                if(mesg[k] < target[k]) {
                    prof[CLEN] = 1;
                    *nonc = le64toh(((uint64_t *)pmesg)[0]);
                    return gs;
                } else if(mesg[k] > target[k]) {
                    break;
                }
            }
        }

    }
  }

  return 200;
}