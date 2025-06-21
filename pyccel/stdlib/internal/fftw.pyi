#$ header metavar external=False
from typing import Final

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
    ...

def fftw_plan_dft_2d(n0 : int, n1 : int, *in : fftw_complex, *out : fftw_complex, sign : int, flags : unsigned):
    ...

def fftw_plan_dft_3d(n0 : int, n1 : int, n2 : int, *in : fftw_complex, *out : fftw_complex, sign : int, flags : unsigned):
    ...

def fftw_plan_dft(rank : int, *n : Final[int],*in : fftw_complex, *out : fftw_complex, sign : int, flags : unsigned):
    ...


def fftw_plan_dft_r2c_1d(n : int, *in : float, *out : fftw_complex, flags : unsigned):
    ...

def fftw_plan_dft_r2c_2d(n0 : int, n1 : int, *in : float, *out : fftw_complex, flags : unsigned):
    ...

def fftw_plan_dft_r2c_3d(n0 : int, n1 : int, n2 : int, *in : float, *out : fftw_complex, flags : unsigned):
    ...

def fftw_plan_dft_r2c(rank : int, *n : Final[int], *in : float, *out : fftw_complex, flags : unsigned):
    ...


def fftw_plan_dft_c2r_1d(n0 : int,*in : fftw_complex, *out : float,flags : unsigned):
    ...

def fftw_plan_dft_c2r_2d(n0 : int, n1 : int,*in : fftw_complex, *out : float,flags : unsigned):
    ...

def fftw_plan_dft_c2r_3d(n0 : int, n1 : int, n2 : int,*in : fftw_complex, *out : float,flags : unsigned):
    ...

def fftw_plan_dft_c2r(rank : int, *n : Final[int],*in : fftw_complex, *out : float,flags : unsigned):
    ...


def fftw_plan_r2r_1d(n : int, *in : float, *out : float, kind : fftw_r2r_kind, flags : unsigned):
    ...

def fftw_plan_r2r_2d(n0 : int, n1 : int, *in : float, *out : float, kind0 : fftw_r2r_kind, kind1 : fftw_r2r_kind, flags : unsigned):
    ...

def fftw_plan_r2r_3d(n0 : int, n1 : int, n2 : int, *in : float, *out : float, kind0 : fftw_r2r_kind, kind1 : fftw_r2r_kind, kind2 : fftw_r2r_kind, flags : unsigned):
    ...

def fftw_plan_r2r(rank : int, *n : Final[int], *in : float, *out : float, *kind : Final[fftw_r2r_kind], flags : unsigned):
    ...



def fftw_plan_many_dft(rank : int, *n : Final[int], howmany : int, *in : fftw_complex, *inembed : Final[int], istride : int, idist : int, *out : fftw_complex, *onembed : Final[int], ostride : int, odist : int,sign : int, flags : unsigned):
    ...


def fftw_plan_many_dft_r2c(rank : int, *n : Final[int], howmany : int,*in : float, *inembed : Final[int], istride : int, idist : int, *out : fftw_complex, *onembed : Final[int], ostride : int, odist : int, flags : unsigned):
    ...


def fftw_plan_many_dft_c2r(rank : int, *n : Final[int], howmany : int,*in : fftw_complex, *inembed : Final[int],istride : int, idist : int,*out : float, *onembed : Final[int], ostride : int, odist : int, flags : unsigned):
    ...


def fftw_plan_many_r2r(rank : int, *n : Final[int], howmany : int,*in : float, *inembed : Final[int], istride : int, idist : int,*out : float, *onembed : Final[int],ostride : int, odist : int,  *kind : Final[fftw_r2r_kind], flags : unsigned):
    ...





def fftw_execute(p):
    ...

def fftw_destroy_plan(p):
    ...

def fftw_free(in):
    ...
