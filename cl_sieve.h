
// cl_sieve.h

typedef struct {

	uint64_t pmin = 0, pmax = 0;
	uint32_t kmin = 0, kmax = 0;	// note k is 32 bit!
	uint32_t nmin = 0, nmax = 0;
	uint32_t nstep;
	uint32_t mont_nstep;
	uint32_t kernel_nstep;
	int32_t bbits;
	uint64_t r0;
	int32_t bbits1;
	uint64_t r1;
	uint64_t lastN;
	bool cw = false;
	bool test = false;
	uint64_t checksum = 0;
	bool compute = false;
	int computeunits;
	uint64_t primecount = 0;
	uint64_t factorcount = 0;

	uint64_t workunit;
	uint64_t p;	// current p
	bool write_state_a_next = true;
	uint64_t last_trickle;

//	tpsieve option -M2, change K's modulus to 2
	uint32_t kstep = 2;
	uint32_t koffset = 1;
//	default for twin prime search
//	uint32_t kstep = 6;
//	uint32_t koffset = 3;


}searchData;

void cl_sieve( sclHard hardware, searchData & sd );

void run_test( sclHard hardware, searchData & sd );
