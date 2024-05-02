from cmd import Cmd
from termcolor import colored

from pyccel.parser import Parser

# ...
def pyccel_parse(code):
    pyccel = Parser(code)
    pyccel.parse()
    return pyccel.ast[0]
# ...



HEADER = """
Pyccel 0.9.1 (default, Nov 23 2017, 16:37:01)

IPyccel 0.0.1 -- An enhanced Interactive Pyccel.
?         -> Introduction and overview of IPyccel's features.
%quickref -> Quick reference.
help      -> Pyccel's own help system.
object?   -> Details about 'object', use 'object??' for extra details.
"""

prompt_in = lambda x: colored('In [{}] '.format(x), 'blue', attrs=['bold'])
prompt_out = lambda x: colored('Out [{}] '.format(x), 'red', attrs=['bold'])

class IPyccel(Cmd):
    intro = HEADER
    i_line = 0
    prompt = prompt_in(i_line)

    # ... redefinition
    def precmd(self, line):
        self.i_line += 1
        self.prompt = prompt_in(self.i_line)
        return super(IPyccel, self).precmd(line)
    # ...

    def do_let(self, *args):
        """."""
        if len(args) == 0:
            raise ValueError('Expecting arguments')

        print ("{}".format(*args))

    def do_quit(self, args):
        """Quits the program."""
        print ("Quitting.")
        raise SystemExit

    def default(self, line):
        """this method will catch all commands. """
        # Assign
        try:
            expr = pyccel_parse(line)
            prefix = prompt_out(self.i_line)
            print('{} {}'.format(prefix, expr))
        except:
            print('Wrong statement. Leaving!!')
            raise SystemExit


def ipyccel():
    IPyccel().cmdloop()
