import sys
import re
import json
import argparse

def     mini_md_summary(title, outcome, c, f, py):
    md = f"## {title} - {outcome} "
    if outcome == "failure":
        if len(c) != 0:
            md = md + '\n' + "### C Test summary: "
            md = md + '\n'
            for i in c:
                md = md + i + "\n"
        if len(f) != 0:
            md = md + '\n' + "### C Test summary: "
            md = md + '\n'
            for i in f:
                md = md + i + "\n"
        if len(py) != 0:
            md = md + '\n' + "### Python Test summary: "
            md = md + '\n'
            for i in py:
                md = md + i + "\n"
    md = md + "\n"
    return(md)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--title', nargs='?', required=True, help='Tests summary title')
    parser.add_argument('-test', '--tests', nargs='+', help="'Test title':outcome{success, failure, cancelled, or skipped}:output_file")
    p_args = parser.parse_args()

    outfile = ""
    args = sys.argv[1:]
    output_file = 'test_json_result.json'
    summary = ""

    for i in p_args.tests:
        values = i.split(':')
        mini_title = values[0] if len(values) >= 1 else None
        outcome = values[1] if len(values) >= 2 else None
        out_file = values[2] if len(values) >= 3 and values[2] != '' else None

        if out_file != None:
            with open(out_file , 'r') as f:
                outfile = f.read()
            
            c_tests = []
            f_tests = []
            py_tests = []
            failed_pattern = r".*FAILED.*"
            c_pattern = r".*\[c\].*"
            f_pattern = r".*\[fortran\].*"
            py_pattern = r".*\[python\].*"

            failed_matches = re.findall(failed_pattern, outfile, re.MULTILINE)
            failed_matches = [re.sub(r'.*FAILED ', "- ``", string) for string in failed_matches]
            

            r = re.compile(c_pattern)
            c_failed = list(filter(r.match, failed_matches))
            c_failed = [re.sub(r'\[c\]', "`` :heavy_multiplication_x:", string) for string in c_failed]

            r = re.compile(f_pattern)
            f_failed = list(filter(r.match, failed_matches))
            f_failed = [re.sub(r'\[fortran\]', "`` :heavy_multiplication_x:", string) for string in f_failed]

            r = re.compile(py_pattern)
            py_failed = list(filter(r.match, failed_matches))
            py_failed = [re.sub(r'\[python\]', "`` :heavy_multiplication_x:", string) for string in py_failed]


        summary = summary + mini_md_summary(mini_title, outcome, c_failed, f_failed, py_failed)
    
    print(summary)
    json_ouput = {
        "title":p_args.title,
        "summary":summary
    }

    with open(output_file, 'w') as f:
        json.dump(json_ouput,f)
