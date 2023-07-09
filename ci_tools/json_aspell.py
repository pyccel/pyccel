import sys
import re
import json
import argparse
import os

def     mini_md_summary(words):
    md = f"## There are misspelled words\n"
    for file_info, word in words:
        md = md + f"### File: {file_info} \n"
        md = md + f"- Misspelled Word: {word} \n"
    return(md)



def extract_misspelled_words(output):
    misspelled_words = []
    pattern = r"Misspelled words:\n<[^>]+>\s(.*?):.*?\n-{80}\n(.*?)\n\n"

    matches = re.findall(pattern, output, re.DOTALL)
    for match in matches:
        file_info = match[0]
        words = match[1].strip().split('\n')

        for word in words:
            word = word.strip()
            if word and not word.startswith('-'):
                misspelled_words.append((file_info, word))

    return misspelled_words

def find_word(file_path, search_word):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        for line_number, line in enumerate(lines, start=1):
            match = re.search(r"\b" + re.escape(search_word) + r"\b", line)
            if match:
                column = match.start() + 1
                return line_number, column

    return None

def     annotations_builder(words):
    annotations = []

    for file_info, word in words:
        line_no, column = find_word("../"+file_info, word) 
        annotation_1 = {
            "path": file_info,
            "start_line": line_no,
            "end_line": line_no,
            "start_column": column,
            "end_column": column + len(word),
            "annotation_level": "warning",
            "message": "Misspelled word ",
            "title": "Aspeel misspelled word"
        }
        annotations.append(annotation_1)
    return annotations


if __name__ == '__main__':

    outfile = ""
    file_output = sys.argv[1]
    output_file = 'test_json_result.json'
    summary = ""
    md = ""

    f = open(file_output, "r")
    misspelled_words = extract_misspelled_words(f.read())
    md = mini_md_summary(misspelled_words)
    annotations = annotations_builder(misspelled_words)
    json_ouput = {
        "title":"Misspelling summary ",
        "summary":md,
        "annotations": annotations
    }

    with open(output_file, 'w') as f:
        json.dump(json_ouput,f)
