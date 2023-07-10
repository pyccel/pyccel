import sys
import re
import json
import argparse
import os

def     mini_md_summary(list_fw):
    md = f"## There are misspelled words\n"
    for file_info, words in list_fw.items():
        md = md + f"### File: {file_info} \n"
        for word in words:
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

def find_all_words(file_path, search_word):
    results = []
    
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
        for line_number, line in enumerate(lines, start=1):
            matches = re.finditer(r"\b" + re.escape(search_word) + r"\b", line)
            
            for match in matches:
                column = match.start() + 1
                results.append((line_number, column))
    
    if results:
        return results
    else:
        return None

def     annotations_builder(words):
    annotations = []

    for file_info, words in words.items():
        for word in words:
            words_list = find_all_words(file_info, word)
            for line_no, column in words_list:
                annotation_1 = {
                    "path": file_info,
                    "start_line": line_no,
                    "end_line": line_no,
                    "start_column": column,
                    "end_column": column + len(word),
                    "annotation_level": "failure",
                    "message": f"Misspelled word {word}",
                    "title": "Misspelled word"
                }
                annotations.append(annotation_1)
    return annotations

def factor_tuples(tuples_list):
    factors = {}
    
    for file_path, word in tuples_list:
        if file_path not in factors:
            factors[file_path] = []
        if word not in factors[file_path]:
            factors[file_path].append(word)
    
    return factors

if __name__ == '__main__':

    outfile = ""
    file_output = sys.argv[1]
    output_file = 'test_json_result.json'
    summary = ""
    md = ""

    with open(file_output, "r") as f:
        misspelled_words_bf = extract_misspelled_words(f.read())
        misspelled_words = factor_tuples(misspelled_words_bf)
        md = mini_md_summary(misspelled_words)
        annotations = annotations_builder(misspelled_words)
        json_ouput = {
            "title":"Misspelling summary ",
            "summary":md,
            "annotations": annotations
        }
    with open(output_file, 'w') as f:
        json.dump(json_ouput,f)
