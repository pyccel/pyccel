
GREEN="\033[1;32m"
NOCOLOR="\033[0m"

python setup.py build
python setup.py install

echo  "${GREEN}"
python test_lists.py
echo "${NOCOLOR}"
