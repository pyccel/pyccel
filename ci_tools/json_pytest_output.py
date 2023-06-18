import sys
import re
import json

if __name__ == '__main__':

    if len(sys.argv) == 1:
        raise ValueError("please provide an output file")

    outfile = ""
    args = sys.argv[1:]
    output_file = 'test_json_result.json'


    for i in sys.argv:
        with open(i , 'r') as f:
            outfile =  outfile + "\n"  + f.read()

    c_tests = []
    f_tests = []
    py_tests = []
    failed_pattern = r"^FAILED.*"
    c_pattern = r".*\[c\].*"
    f_pattern = r".*\[fortran\].*"
    py_pattern = r".*\[python\].*"

    failed_matches = re.findall(failed_pattern, outfile, re.MULTILINE)

    r = re.compile(c_pattern)
    c_failed = list(filter(r.match, failed_matches))

    r = re.compile(f_pattern)
    f_failed = list(filter(r.match, failed_matches))

    r = re.compile(py_pattern)
    py_failed = list(filter(r.match, failed_matches))

    json_ouput = {
        "title":"linux unit test",
        "summary":
        {
            "c tests":c_failed,
            "fortran tests":f_failed,
            "python tests":py_failed
        }
    }

    json_f = json.dump(json_ouput)
    with open(output_file, 'a') as f:
        print(json_f, sep='', file=f)

