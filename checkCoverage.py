import os

parser = argparse.ArgumentParser(description='Coverage checker')
parser.add_argument('MinPercent', metavar='min',nargs=1,type=int,
                   help='Minimum percentage of coverage required for a pass')
args = parser.parse_args()

file = open("cov_out.txt", "r")
out = file.read()
file.close()
os.remove("cov_out.txt")
out = out[2:-1]
out = out.replace("\\n","\n")
print(out)

file = open("cov_err.txt", "r")
err = file.read()
file.close()
os.remove("cov_err.txt")
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
