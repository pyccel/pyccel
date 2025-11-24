#$ header metavar includes="__pyccel__mod__"
#$ header metavar libraries="final_destroy"
#$ header metavar libdirs="."

class T:
    @low_level('init')
    def __init__(self): ...
