""" Script for removing unnecessary `@types` decorators
"""
import os
import re
import sys
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser(description='Tool for removing unnecessary `@types` decorators')

    parser.add_argument('folder', type=str, help='The folder in which the @types decorator will be looked for')

    args = parser.parse_args()

    type_annotation = re.compile(r"'[a-zA-Z0-9\[:,\] ]*'")

    for dirpath, dirnames, filenames in os.walk(args.folder):
        for name in filenames:
            filename = os.path.join(dirpath, name)
            if not name.endswith('.py') or os.path.abspath(filename) == __file__ or '__pycache__' in dirpath:
                continue
            print("Treating",filename)
            with open(filename, 'r') as file:
                lines = file.readlines()

            n = len(lines)
            i = 0

            current_types = []
            new_lines = []
            uses_types = False
            while i<n:
                if '@types' in lines[i]:
                    assert ')' in lines[i]
                    current_types.append(lines[i])
                elif 'def ' in lines[i]:
                    if len(current_types) == 1:
                        type_strs = current_types[0].split('(')[1].split(')')[0]
                        quoted_types = [q.strip() for q in type_annotation.findall(type_strs)]
                        type_strs = type_annotation.sub('None', type_strs)
                        all_types = []
                        j = 0
                        for s in type_strs.split(','):
                            s = s.strip()
                            if s == 'None':
                                all_types.append(quoted_types[j])
                                j+=1
                            else:
                                all_types.append(s)
                        type_annotations = [s.replace('real', 'float').strip() for s in all_types]
                        start, tmp = lines[i].split('(')
                        args, end = tmp.split(')')
                        arguments = [a.strip() for a in args.split(',')]
                        if len(arguments) == 1 and arguments[0] == '':
                            new_lines.append(f"{start}(){end}")
                        else:
                            default_args = [a.split('=') for a in arguments]
                            arg_dict = {da[0] : "" if '=' not in a else f" = {da[1]}" for a,da in zip(arguments, default_args)}
                            new_args = [f"{key} : {annot}{default}" for (key,default), annot in zip(arg_dict.items(), type_annotations)]
                            new_lines.append(f"{start}({', '.join(new_args)}){end}")
                    else:
                        new_lines.extend(current_types)
                        new_lines.append(lines[i])
                        uses_types = (len(current_types) != 0)
                    current_types = []
                elif '@' in lines[i]:
                    if '(' in lines[i]:
                        while ')' not in lines[i]:
                            new_lines.append(lines[i])
                            i+=1
                    new_lines.append(lines[i])
                elif 'from pyccel.decorators import' in lines[i] and 'types' in lines[i]:
                    import_types_line = i
                    if ',' in lines[i]:
                        imps = [i.strip() for i in lines[i].split('from pyccel.decorators import')[1].split(',')]
                        new_lines.append('from pyccel.decorators import '+', '.join(i for  i in imps if i != 'types')+'\n')
                else:
                    assert len(current_types) == 0
                    new_lines.append(lines[i])
                i+=1

            if uses_types:
                new_lines.insert(import_types_line, "from pyccel.decorators import types\n")

            with open(filename, 'w') as file:
                file.write(''.join(new_lines))

