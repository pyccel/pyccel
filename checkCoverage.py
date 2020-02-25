import argparse

parser = argparse.ArgumentParser(description='Coverage checker')
parser.add_argument('min', metavar='MinPercent',nargs=1,type=int,
                   help='Minimum percentage of coverage required for a pass')
args = parser.parse_args()

file = open("cov_out.txt", "r")
out = file.read()
file.close()
out = out[2:-1]
out = out.replace("\\n","\n")
print(out)

file = open("cov_err.txt", "r")
err = file.read()
file.close()
err = err[2:-1]
err = err.replace("\\n","\n")
print(err)

i = out.find("TOTAL")
if ( i == -1 ):
    assert(False)
else:
    totline = out[i:]
    pc=float(totline[totline.rfind(" "):totline.rfind("%")])
    assert(pc>=args.min[0])
