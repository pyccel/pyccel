
# pylint: disable=missing-function-docstring, missing-module-docstring
def create_list():
    a = [1,2,3]# pylint: disable=unused-variable

def create_set():
    a = {1,2,3}# pylint: disable=unused-variable

def create_dict():
    a = {1:1,2:2,3:3}# pylint: disable=unused-variable

def list_reassign():
    a = [1,2,3]# pylint: disable=unused-variable
    a = [1,2,3,4]# pylint: disable=unused-variable
    a = [i for i in range(15)]# pylint: disable=unused-variable
    a = [1,2]# pylint: disable=unused-variable

def set_reassign():
    a = {1,2,3}# pylint: disable=unused-variable
    a = {1,2}# pylint: disable=unused-variable
    a = {1,2,3,4}# pylint: disable=unused-variable
    a = {1,2,3,4,5}# pylint: disable=unused-variable

def dict_reassign():
    a = {1:1,2:2,3:3}# pylint: disable=unused-variable
    a = {1:1,2:2,3:3,4:4}# pylint: disable=unused-variable
    a = {1:1,2:2}# pylint: disable=unused-variable
    a = {1:1,2:2,3:3,4:4,5:5}# pylint: disable=unused-variable

def conditional_list(b1: bool):
    if (b1):
        a = [1,2,3]# pylint: disable=unused-variable
    a = [1,2,3,4]# pylint: disable=unused-variable

def conditional_set(b1: bool):
    if (b1):
        a = {1,2,3}# pylint: disable=unused-variable
    a = {1,2,3,4}# pylint: disable=unused-variable

def conditional_dict(b1: bool):
    if (b1):
        a = {1:1,2:2,3:3}# pylint: disable=unused-variable
    a = {1:1,2:2,3:3,4:4,5:5}# pylint: disable=unused-variable

if __name__ == '__main__':
    create_list()
    create_set()
    create_dict()
    list_reassign()
    set_reassign()
    dict_reassign()
    conditional_list(True)
    conditional_set(True)
    conditional_dict(True)
    conditional_list(False)
    conditional_set(False)
    conditional_dict(False)
