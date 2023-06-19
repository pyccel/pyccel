import sys
import re
import json

def     md_summary(c, f, py):
    md = "## C Tests summary: "
    if len(c) == 0:
        md = md + "\n All Tests PASSED ✅\n"
    else:
        md = md + '\n'
        for i in c:
            md = md + i + "\n"
    md += "## FORTRAN Tests Summary: "
    if len(f) == 0:
        md = md + "\n All Tests PASSED ✅\n"
    else:
        md = md + '\n'
        for i in f:
            md = md + i + "\n"
    md += "## PYTHON Tests Summary: "
    if len(f) == 0:
        md = md + "\n All Tests PASSED ✅\n"
    else:
        md = md + '\n'
        for i in py:
            md = md + i + "\n"
    return(md)

if __name__ == '__main__':

    if len(sys.argv) == 1:
        raise ValueError("please provide an output file")

    outfile = ""
    args = sys.argv[1:]
    output_file = 'test_json_result.json'


    for i in args:
        with open(i , 'r') as f:
            outfile =  outfile + "\n"  + f.read()

    c_tests = []
    f_tests = []
    py_tests = []
    failed_pattern = r".*FAILED.*"
    c_pattern = r".*\[c\].*"
    f_pattern = r".*\[fortran\].*"
    py_pattern = r".*\[python\].*"

    failed_matches = re.findall(failed_pattern, outfile, re.MULTILINE)
    #print(f"--------------------------------\n{failed_matches}")
    failed_matches = [re.sub(r'.*FAILED ', "- ", string) for string in failed_matches]
    

    r = re.compile(c_pattern)
    c_failed = list(filter(r.match, failed_matches))
    c_failed = [re.sub(r'\[c\]', " ❌", string) for string in c_failed]

    r = re.compile(f_pattern)
    f_failed = list(filter(r.match, failed_matches))
    f_failed = [re.sub(r'\[fortran\]', " ❌", string) for string in f_failed]

    r = re.compile(py_pattern)
    py_failed = list(filter(r.match, failed_matches))
    py_failed = [re.sub(r'\[python\]', " ❌", string) for string in py_failed]


    summary = md_summary(c_failed, f_failed, py_failed)
    print(summary)
    json_ouput = {
        "title":"linux unit test",
        "summary":summary
    }

    with open(output_file, 'w') as f:
        json.dump(json_ouput,f)
