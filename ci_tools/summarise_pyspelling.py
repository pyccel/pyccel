""" Script to output pyspelling results in a more digestible markdown format
"""
import argparse
import difflib
import os
import sys
import json
import re


def find_all_words(file_path, search_word):
    """
    Find all occurrences of a word in a file.

    Find all occurrences of a word in a file and return the line number and column of each occurrence of the search word.

    Parameters
    ----------
    file_path : str
        The path to the file to search.
    search_word : str
        The word to search for.

    Returns
    -------
    list
        A list of tuples, where each tuple contains the line number and
        column number of an occurrence of the search word.
    """
    results = []
    regex = re.compile(r"\b" + re.escape(search_word) + r"\b")

    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()
        for line_number, line in enumerate(lines, start=1):
            matches = regex.finditer(line)
            for match in matches:
                column = match.start() + 1
                results.append((line_number, column))

    if results:
        return results
    return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a neat markdown file to summarise the results')
    parser.add_argument('spelling', metavar='diffFile', type=str,
                            help='File containing the pyspelling output')
    parser.add_argument('output', metavar='output', type=str,
                            help='File where the markdown output will be printed')

    args = parser.parse_args()

    with open(args.spelling, encoding="utf-8") as f:
        lines = f.readlines()

    lines = [l.strip() for l in lines[:-1]]
    lines = [l for l in lines if l != 'Misspelled words:' and any(c != '-' for c in l)]

    errors = {}
    annotations = []

    n = len(lines)
    i = 0
    while i < n:
        filename = lines[i].split()[1]
        i+=1
        words = set()
        while i<n and not lines[i].startswith('<htmlcontent>'):
            words.add(lines[i])
            i+=1

        if filename in errors:
            errors[filename].update(words)
        else:
            errors[filename] = words

    if errors:
        all_words = set()

        with open(os.path.join(os.path.dirname(__file__),'..','.dict_custom.txt'), encoding="utf-8") as d:
            internal_dict = [w.strip() for w in d.readlines()]

        with open(args.output, 'w', encoding="utf-8") as f:
            print("There are misspelled words", file=f)
            for name, words in errors.items():
                print("## `", name, "`", file=f)
                for w in words:
                    words_list = find_all_words(os.path.join("..",name.strip(":")), w)
                    suggestions = difflib.get_close_matches(w, internal_dict)
                    for line_no, column in words_list:
                        if suggestions:
                            msg_cus = f" Misspelled word :  Did you mean {w} -> {suggestions}"
                        else:
                            msg_cus = f"Misspelled word {w}"
                        annotation_1 = {
                            "path": name,
                            "start_line": line_no,
                            "end_line": line_no,
                            "start_column": column,
                            "end_column": column + len(w),
                            "annotation_level": "failure",
                            "message": msg_cus,
                            "title": "Misspelled word"
                        }
                        annotations.append(annotation_1)
                    if suggestions:
                        print("-   ", w, f"  :  Did you mean {w} -> {suggestions}", file=f)
                    else:
                        print("-   ", w, file=f)
                print(file=f)
                all_words.update(words)

            print("These errors may be due to typos, capitalisation errors, or lack of quotes around code. If this is a false positive please add your word to `.dict_custom.txt`", file=f)

        # Generating a json file for github check runs
        output_file = 'test_json_result.json'
        md = ""

        with open(args.output, 'r', encoding="utf-8") as f:
            md = f.read()
        print(md)
        json_ouput = {
            "title":"Misspelling summary ",
            "summary":md,
            "annotations": annotations
        }
        with open(output_file, 'w', encoding="utf-8") as f:
            json.dump(json_ouput,f)
        sys.exit(1)
    else:
        sys.exit(0)
