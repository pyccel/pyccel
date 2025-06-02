#$ header metavar external=False

FFTW_FORWARD : int
FFTW_BACKWARD : int
FFTW_REAL_TO_COMPLEX : int
FFTW_COMPLEX_TO_REAL : int
FFTW_ESTIMATE : int
FFTW_MEASURE : int
FFTW_OUT_OF_PLACE : int
FFTW_IN_PLACE : int
FFTW_USE_WISDOM : int
FFTW_THREADSAFE : int

def fftw_plan_dft_1d(int, *in : complex, *out : complex, sign : int, flags : unsigned):
    pass

def fftw_plan_dft_2d(n0 : int, n1 : int, fftw_*in : complex, fftw_*out : complex, sign : int, flags : unsigned):
    pass

def fftw_plan_dft_3d(n0 : int, n1 : int, n2 : int, fftw_*in : complex, fftw_*out : complex, sign : int, flags : unsigned):
    pass

def fftw_plan_dft(rank : int, *n : 'const int',fftw_*in : complex, fftw_*out : complex, sign : int, flags : unsigned):
    pass


def fftw_plan_dft_r2c_1d(n : int, *in : float, fftw_*out : complex, flags : unsigned):
    pass

def fftw_plan_dft_r2c_2d(n0 : int, n1 : int, *in : float, fftw_*out : complex, flags : unsigned):
    pass

def fftw_plan_dft_r2c_3d(n0 : int, n1 : int, n2 : int, *in : float, fftw_*out : complex, flags : unsigned):
    pass

def fftw_plan_dft_r2c(rank : int, *n : 'const int', *in : float, fftw_*out : complex, flags : unsigned):
    pass


def fftw_plan_dft_c2r_1d(n0 : int,fftw_*in : complex, *out : float,flags : unsigned):
    pass

def fftw_plan_dft_c2r_2d(n0 : int, n1 : int,fftw_*in : complex, *out : float,flags : unsigned):
    pass

def fftw_plan_dft_c2r_3d(n0 : int, n1 : int, n2 : int,fftw_*in : complex, *out : float,flags : unsigned):
    pass

def fftw_plan_dft_c2r(rank : int, *n : 'const int',fftw_*in : complex, *out : float,flags : unsigned):
    pass


def fftw_plan_r2r_1d(n : int, *in : float, *out : float, fftw_r2r_kind kind, flags : unsigned):
    pass

def fftw_plan_r2r_2d(n0 : int, n1 : int, *in : float, *out : float, fftw_r2r_kind kind0, fftw_r2r_kind kind1, flags : unsigned):
    pass

def fftw_plan_r2r_3d(n0 : int, n1 : int, n2 : int, *in : float, *out : float, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, flags : unsigned):
    pass

def fftw_plan_r2r(rank : int, *n : 'const int', *in : float, *out : float, const fftw_r2r_kind *kind, flags : unsigned):
    pass



def fftw_plan_many_dft(rank : int, *n : 'const int', howmany : int, fftw_*in : complex, *inembed : 'const int', istride : int, idist : int, fftw_*out : complex, *onembed : 'const int', ostride : int, odist : int,sign : int, flags : unsigned):
    pass


def fftw_plan_many_dft_r2c(rank : int, *n : 'const int', howmany : int,*in : float, *inembed : 'const int', istride : int, idist : int, fftw_*out : complex, *onembed : 'const int', ostride : int, odist : int, flags : unsigned):
    pass


def fftw_plan_many_dft_c2r(rank : int, *n : 'const int', howmany : int,fftw_*in : complex, *inembed : 'const int',istride : int, idist : int,*out : float, *onembed : 'const int', ostride : int, odist : int, flags : unsigned):
    pass


def fftw_plan_many_r2r(rank : int, *n : 'const int', howmany : int,*in : float, *inembed : 'const int', istride : int, idist : int,*out : float, *onembed : 'const int',ostride : int, odist : int, const fftw_r2r_kind *kind, flags : unsigned):
    pass





def fftw_execute(p)
def fftw_destroy_plan(p)
def fftw_free(in):
    pass
