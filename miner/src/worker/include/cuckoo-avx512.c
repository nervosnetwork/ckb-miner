#include <immintrin.h>
#include <stdint.h>    // for types uint32_t,uint64_t
#include "portable_endian.h"    // for htole32/64

#include <stdlib.h>
#include <string.h>
#include "blake2b.h"

#define EBIT 15
#define CLEN 12

#define ROL(x, b)   _mm512_rol_epi64 ((x), (b))
#define SL(x, b)    _mm512_slli_epi64((x), (b))
#define AND(a, b)   _mm512_and_epi64 ((a), (b))
#define XOR(a, b)   _mm512_xor_epi64 ((a), (b))
#define OR(a, b)    _mm512_or_epi64 ((a), (b))
#define ADD(a, b)   _mm512_add_epi64((a), (b))
#define SET(a)      _mm512_set1_epi64((a))
#define STORE       _mm512_store_epi64
#define SET8        _mm512_set_epi64
#define u512        __m512i

#define EN 1 << EBIT
#define CN CLEN << 2
#define M EN << 1
#define MASK (1 << EBIT) - 1

// set siphash keys from 32 byte char array
#define setkeys() \
  k0 = SET(le64toh(((uint64_t *)mesg)[0])); \
  k1 = SET(le64toh(((uint64_t *)mesg)[1])); \
  k2 = SET(le64toh(((uint64_t *)mesg)[2])); \
  k3 = SET(le64toh(((uint64_t *)mesg)[3]));


#define sip_round() \
  v0 = ADD(v0,v1); v2 = ADD(v2,v3); v1 = ROL(v1,13); \
  v3 = ROL(v3,16); v1 = XOR(v1,v0); v3 = XOR(v3,v2); \
  v0 = ROL(v0,32); v2 = ADD(v2, v1); v0 = ADD(v0, v3); \
  v1 = ROL(v1,17); v3 = ROL(v3,21); \
  v1 = XOR(v1,v2); v3 = XOR(v3,v0); v2 = ROL(v2,32); 

#define siphash24() \
  v0 = k0; v1 = k1; v2 = k2; v3 = k3; \
  v3 = XOR(v3,nonce); \
  sip_round(); sip_round(); \
  v0 = XOR(v0,nonce); \
  v2 = XOR(v2,k4); \
  sip_round(); sip_round(); sip_round(); sip_round(); \
  h = OR((SL(AND((XOR(XOR(XOR(v0,v1),v2),v3)),mask), 1)), flag); 
  

int c_solve_avx(uint32_t *prof, uint64_t *nonc, const uint8_t *hash, const uint8_t *target) {
  HCRYPTPROV Rnd;
  int graph[M];
  uint64_t *G = _mm_malloc(sizeof(uint64_t) * M, 64);
  int path[CLEN];

  uint8_t pmesg[40];
  uint8_t hmesg[CN];
  uint8_t mesg[32];

  blake2b_state S;

  u512 k0, k1, k2, k3, k4;
  u512 v0, v1, v2, v3, nonce, mask, flag;
  u512 h;
  uint64_t e3,e2,e1,e0;
  
  k4 = SET(0xff);
  mask = SET(MASK);
  flag = SET8(1,0,1,0,1,0,1,0);

  b2b_setup(&S);
  
  memcpy(pmesg+8, hash, 32);
  CryptGenRandom(Rnd, 8, pmesg);
  
  for(uint64_t gs=1; gs<300; ++gs) {
    ((uint64_t *)pmesg)[0] = ((uint64_t *)pmesg)[0] ^ gs;
    blake2b_state tmp = S;
    b2b_update(&tmp, pmesg, 40);
    b2b_final(&tmp, mesg, 32);

    setkeys();
    
    for(int i=0; i<M; ++i) {
        graph[i] = -1;
    }

    for(uint64_t i=0, j=0; i<EN; j+=8) {
        e0 = i; ++i; e1 = i; ++i;
        e2 = i; ++i; e3 = i; ++i;
        nonce = OR(SL(SET8(e3,e3,e2,e2,e1,e1,e0,e0),1), flag);
        siphash24();
        STORE(G+j, h);
    }

    for(uint64_t i=0; i<M;) {
        int u = G[i]; ++i;
        int v = G[i]; ++i;

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
            for(j=0; k < b; ) {
                int u = G[j]; ++j;
                int v = G[j]; ++j;

                if (graph[u] == v) {
                    prof[k] = (j >> 1) - 1;
                    graph[u] = -1;
                    ++k;
                } else if (graph[v] == u) {
                    prof[k] = (j >> 1) - 1;
                    graph[v] = -1;
                    ++k;
                }
            }

            prof[k] = (i >> 1) -1;
            
            memcpy(hmesg, prof, CN);
            blake2b_state tmp = S;
            b2b_update(&tmp, hmesg, CN);
            b2b_final(&tmp, mesg, 32);
            
            for(int k=0; k<32; ++k) {
                if(mesg[k] < target[k]) {
                    prof[CLEN] = 1;
                    *nonc = le64toh(((uint64_t *)pmesg)[0]);
                    _mm_free(G);
                    return gs;
                } else if(mesg[k] > target[k]) {
                    break;
                }
            }
        }
    }
  }
  _mm_free(G);
  prof[CLEN] = 0;
  return 300;
}