# coding: utf8
#
# Author: Yaman Güçlü, Jan 2016 - IPP Garching
#
# Last revision: 2 Jun 2016
#


#===============================================================================

def make_package( input_libraries, output_package ):
    """
    Repackage contents of multiple static libraries into a single archive.

    input_libraries: list
        list of input files

    output_package: string
        path to the contructed library
    """

    import os, shutil

    # Collect absolute paths to all library files
    input_libraries = [os.path.abspath( f ) for f in input_libraries]
    output_package  =  os.path.abspath( output_package )

    # Determine package name and directory, and create temporary directory
    PKG_DIR, PKG_NAME = os.path.split( output_package )
    TMP_DIR = os.path.join( os.path.abspath( os.path.curdir ), "tmp__object_files" )

    if os.path.isdir( TMP_DIR ):
        shutil.rmtree( TMP_DIR )

    os.mkdir( TMP_DIR )
    os.chdir( TMP_DIR )

    # Extract all object files into temporary directory
    for lib in input_libraries:
        command = 'ar -x "{:s}"'.format( lib )
        exit_status = os.system( command )
        assert exit_status == 0

    # Repackage all object files into a single archive (static library)
    command = 'ar -qc "{:s}" *.o'.format( PKG_NAME )
    exit_status = os.system( command )
    assert exit_status == 0

    # remove all libs
    # TODO ARA maybe add some comments in the generated file
    # TODO JLA don't need anymore
#    lines = []
#    for lib in input_libraries:
#        command = lib.split("/")[-1] # only extract the lib name
#        lines.append(command)
#        lines.append("\n")
#    PKG_INPUT_LIBRARIES = "libs.txt"
#    f = open(PKG_INPUT_LIBRARIES, 'w')
#    for line in lines:
#        f.write(line)
#    f.close()

    # Move library to proper build directory, and remove temporary dir
    shutil.copy( PKG_NAME, PKG_DIR )
#    shutil.copy( PKG_INPUT_LIBRARIES, PKG_DIR )
    shutil.rmtree( TMP_DIR )

#===============================================================================

def parse_input():

  import argparse, sys

  parser = argparse.ArgumentParser (
      prog        = 'python ' + sys.argv[0],
      description = 'Repackage contents of multiple static libraries into a new single library.',
      epilog      = ' ',
      formatter_class = argparse.ArgumentDefaultsHelpFormatter,
      )

  parser.add_argument('output_package',
                      metavar = 'pkg',
                      help = 'Relative or absolute path of output archive')

  parser.add_argument('input_libraries',
                      metavar = 'lib',
                      nargs = '+',
                      help = 'Relative or absolute path of input library')

  return parser.parse_args()

#-------------------------------------------------------------------------------
if __name__ == '__main__':
    import os
    args   = parse_input()
    origin = os.path.abspath( os.path.curdir )
    try:
        make_package( args.input_libraries, args.output_package )
    except:
        os.chdir( origin )
        raise
