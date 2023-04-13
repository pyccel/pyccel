# pylint: disable=missing-function-docstring, missing-module-docstring
# this file is used inside imports.py
# make sure that you update the imports.py file if needed
# pylint: disable=unused-variable

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

conversion_1  = int(3.0)
conversion_2  = 2.0 + int(3.0)
conversion_3  = 2.0 - int(3.0)
conversion_4  = 2.0 * int(3.0)
conversion_5  = 2.0 / int(3.0)
conversion_6  = 2.0 % int(3.0)
conversion_7  = float(3.0)
conversion_8  = 2.0 + float(3.0)
conversion_9  = 2.0 - float(3.0)
conversion_10 = 2.0 * float(3.0)
conversion_11 = 2.0 / float(3.0)
conversion_12 = 2.0 % float(3.0)

# this statement does not make sense in a program
x = ad is None

a,b,c = 1, False, 3.4
