
/* 

	verify_factor.c

	Bryan Little 2/12/2023

	Yves Gallot

*/

#include "verify_factor.h"


uint64_t mInvert(uint64_t p)
{
	uint64_t p_inv = 1, prev = 0;
	while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
	return p_inv;
}


uint64_t mMul(uint64_t a, uint64_t b, uint64_t p, uint64_t q)
{
	unsigned __int128 res;

	res  = (unsigned __int128)a * b;
	uint64_t ab0 = (uint64_t)res;
	uint64_t ab1 = res >> 64;

	uint64_t m = ab0 * q;

	res = (unsigned __int128)m * p;
	uint64_t mp = res >> 64;

	uint64_t r = ab1 - mp;

	return ( ab1 < mp ) ? r + p : r;
}


uint64_t mAdd(uint64_t a, uint64_t b, uint64_t p)
{
	uint64_t r;

	uint64_t c = (a >= p - b) ? p : 0;

	r = a + b - c;

	return r;
}


bool verify_factor(	uint64_t p,
			uint64_t k,
			uint32_t n,
			int32_t c	){


	uint64_t q = mInvert(p);
	uint64_t one = (-p) % p;
	uint64_t pmo = p - one;	
	uint64_t two = mAdd(one, one, p);
	uint64_t t = mAdd(two, two, p);
	for (int i = 0; i < 5; ++i)
		t = mMul(t, t, p, q);	// 4^{2^5} = 2^64
	uint64_t r2 = t;


	uint32_t exp = n;
	uint32_t curBit = 0x80000000;
	curBit >>= ( __builtin_clz(exp) + 1 );

	uint64_t a = two;  // 2 in montgomery form

	uint64_t Km = mMul(k,r2,p,q);  // convert k to montgomery form

	// a = 2^n mod P
	while( curBit )
	{
		a = mMul(a,a,p,q);

		if(exp & curBit){
			a = mAdd(a,a,p);
		}

		curBit >>= 1;
	}

	// b = k*2^n mod P
	uint64_t b = mMul(a,Km,p,q);

	if(b == one && c == -1){
//		printf("%" PRIu64 " is a factor of %" PRIu64 "*2^%u-1\n",p,k,n);
		return true;
	}
	else if(b == pmo && c == 1){
//		printf("%" PRIu64 " is a factor of %" PRIu64 "*2^%u+1\n",p,k,n);
		return true;
	}

	return false;

}



