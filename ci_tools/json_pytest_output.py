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

def     mini_md_summary(title, outcome, failed_tests):
    """
    Generate Markdown.

    Generate a Markdown summary based on the provided information.

    Parameters
    ----------
    title : str
        The title of the summary.
    outcome : str
        The result of a completed step, Possible values are success, failure, cancelled, or skipped.
    failed_tests : dict
        A dictionary whose keys are languages and whose values are lists of failed test summaries.

    Returns
    -------
    str
        The Markdown summary string.
    """
    md = f"## {title} - {outcome} "
    if outcome == "failure":
        for lang, errs in failed_tests.items():
            if len(errs) != 0:
                md = md + '\n' + f"### {lang.capitalize()} Test summary: "
                md = md + '\n'
                for i in errs:
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

    failed_pattern = re.compile(r".*FAILED.*")
    languages = ('c', 'fortran', 'python')
    pattern = {lang: re.compile(r".*\["+lang+r"\]\ \_.*") for lang in languages}

    for i in p_args.tests:
        values = i.split(':')
        mini_title = values[0] if len(values) >= 1 else None
        outcome = values[1] if len(values) >= 2 else None
        out_file = values[2] if len(values) >= 3 else None


        fails = {}

        if out_file is not None and os.path.exists(out_file):
            with open(out_file , 'r', encoding='utf-8') as f:
                outfile = f.read()

            c_tests = []
            f_tests = []
            py_tests = []

            failed_matches = failed_pattern.findall(outfile, re.MULTILINE)
            failed_matches = [re.sub(r'.*FAILED ', "- ``", string) for string in failed_matches]

            for lang in languages:
                failed_matches = pattern[lang].findall(outfile, re.MULTILINE)
                failed_matches = ["- ``" + string.strip('_') for string in failed_matches]
                fails[lang] = [re.sub(r'\['+lang+r'\]', "`` :heavy_multiplication_x:", string) for string in failed_matches]

        summary = summary + mini_md_summary(mini_title, outcome, fails)

    print(summary)
    json_ouput = {
        "title":p_args.title,
        "summary":summary
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_ouput,f)
