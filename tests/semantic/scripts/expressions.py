# this file is used inside imports.py
# make sure that you update the imports.py file if needed

ai = 1
bi = ai
ci = ai + 1
di = ci * 2 + 3 * bi + ai
ti = ai + bi * (ci + di)

ad = 1.
bd = ad
cd = ad + 1.
dd = cd * 2. + 3. * bd + ad
td = ad + bd * (cd + dd)

# this statement will be ignored at the codegen
x = ad is None
