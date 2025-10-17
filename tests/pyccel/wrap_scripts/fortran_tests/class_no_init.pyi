#$ header metavar includes="__pyccel__mod__"
#$ header metavar libraries="class_no_init"
#$ header metavar libdirs="."

class MethodCheck:
    my_value : int
    def stash_value(self, val : int): ...
