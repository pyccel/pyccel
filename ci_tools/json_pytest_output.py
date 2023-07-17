"""
Command-line script for generating a test summary and saving it as a JSON file.

This script parses command-line arguments to generate a test summary based on the provided inputs.
It processes test outcomes, then generates a Markdown summary using the 'mini_md_summary' function.
The resulting summary is printed to the console and saved as a JSON file.

Usage:
    -t, --title (str): The title of the test summary.
    -test, --tests (list): Test specifications in the format 'Test title:outcome:output_file'.

Example:
    python script.py -t "Test Summary" --tests "Test 1:success:output.txt" "Test 2:failure:error.txt"
"""
import argparse
import json
import os
import re
import sys

def     mini_md_summary(title, outcome, c, f, py):
    """
    Generate Markdown.

    Generate a Markdown summary based on the provided information.

    Parameters
    ----------
    title : str
        The title of the summary.
    outcome : str
        The result of a completed step, Possible values are success, failure, cancelled, or skipped.
    c : list
        A list of C test summaries, (Failed tests).
    f : list
        A list of Fortran test summaries, (Failed tests).
    py : list
        A list of Python test summaries, (Failed tests).

    Returns
    -------
    str
        The Markdown summary string.
    """
    md = f"## {title} - {outcome} "
    if outcome == "failure":
        if len(c) != 0:
            md = md + '\n' + "### C Test summary: "
            md = md + '\n'
            for i in c:
                md = md + i + "\n"
        if len(f) != 0:
            md = md + '\n' + "### Fortran Test summary: "
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
        out_file = values[2] if len(values) >= 3 else None

        if out_file != None and os.path.exists(out_file):
            with open(out_file , 'r') as f:
                outfile = f.read()
            
            c_tests = []
            f_tests = []
            py_tests = []
            failed_pattern = r".*FAILED.*"
            c_pattern = r".*\[c\].*"
            f_pattern = r".*\[fortran\]\ \_.*"
            py_pattern = r".*\[python\]\ \_.*"

            failed_matches = re.findall(failed_pattern, outfile, re.MULTILINE)
            failed_matches = [re.sub(r'.*FAILED ', "- ``", string) for string in failed_matches]
            

            r = re.compile(c_pattern)
            c_failed = list(filter(r.match, failed_matches))
            c_failed = [re.sub(r'\[c\]', "`` :heavy_multiplication_x:", string) for string in c_failed]

            failed_matches = re.findall(f_pattern, outfile, re.MULTILINE)
            failed_matches = ["- ``" + string.strip('_') for string in failed_matches]
            f_failed = [re.sub(r'\[fortran\]', "`` :heavy_multiplication_x:", string) for string in failed_matches]

            failed_matches = re.findall(py_pattern, outfile, re.MULTILINE)
            failed_matches = ["- ``" + string.strip('_') for string in failed_matches]
            py_failed = [re.sub(r'\[python\]', "`` :heavy_multiplication_x:", string) for string in failed_matches]


        summary = summary + mini_md_summary(mini_title, outcome, c_failed, f_failed, py_failed)
    
    print(summary)
    json_ouput = {
        "title":p_args.title,
        "summary":summary
    }

    with open(output_file, 'w') as f:
        json.dump(json_ouput,f)
