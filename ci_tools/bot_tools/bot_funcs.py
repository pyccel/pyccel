
default_python_versions = {
        'anaconda_linux': '3.10',
        'anaconda_windows': '3.10',
        'coverage': '3.7',
        'doc_coverage': '3.8',
        'lint': '3.8',
        'linux': '3.7',
        'macosx': '3.10',
        'pickle_wheel': '3.7',
        'pickle': '3.8',
        'pyccel_lint': '3.8',
        'spelling': '3.8',
        'windows': '3.8'
        }

def show_tests():
    GitHubAPIInteractions(repo, install_token).create_comment(message_from_file('show_tests.txt'))

def show_commands():
    GitHubAPIInteractions(repo, install_token).create_comment(message_from_file('bot_commands.txt'))

def trust_user():
    pass

def run_tests(, python_version = None):
    pass

def trigger_mark_as_ready():
    pass

def mark_as_ready():
    pass

def accept_coverage():
    pass
