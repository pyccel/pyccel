CLAPP
=====

PLAF
****

run:: 

  pyccel --language="fortran" --compiler=gfortran --filename=tests/clapp/plaf/test_matrix_csr.py --include=$CLAPP_DIR/include/plaf --libdir=$CLAPP_DIR/lib/ --libs=plaf --execute

SPL
***

run:: 

  pyccel --language="fortran" --compiler=gfortran --filename=tests/clapp/plaf/?.py --include=$CLAPP_DIR/include/spl --libdir=$CLAPP_DIR/lib/ --libs=spl --execute

