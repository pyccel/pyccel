#include "cwrapper.h"

/*                                                              */
/*                        CAST_FUNCTIONS                        */
/*                                                              */

// Python to C

float complex	PyComplex_to_Complex64(Pyobject *o)
{
    float			real_part;
    float			imag_part;
    float complex	c;

    real_part = PyComplex_Real_AsDouble(o);
    imag_part = PyComplex_Imag_AsDouble(o);

    c = CMPLXF(real_part, imag_part);
    return c;
}

double complex	PyComplex_to_Complex128(Pyobject *o)
{
    double			real_part;
    double			imag_part;
    double complex	c;

    real_part = PyComplex_Real_AsDouble(o);
    imag_part = PyComplex_Imag_AsDouble(o);

    c = CMPLX(real_part, imag_part);
    return c;
}

bool			PyBool_to_Bool(Pyobject *o)
{
    bool	b;
    b = o == PyTrue;

    return b;
}
