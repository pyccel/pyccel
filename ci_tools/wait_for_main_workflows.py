""" Module which waits for the workflows triggered on the main branch to finish and asserts
that they pass. This is must succeed before the code can be deployed to PyPi.
"""
import os
from bot_tools.github_api_interactions import GitHubAPIInteractions

if __name__ == '__main__':
    GAI = GitHubAPIInteractions(os.environ['GITHUB_REPOSITORY'])
    required_workflows = ('Anaconda-Linux', 'Anaconda-Windows', 'Intel unit tests',
                          'Linux unit tests', 'MacOSX unit tests', 'Windows unit tests',
                          'Installation', 'Wheel-installation')
    assert GAI.wait_for_runs(os.environ['COMMIT'], required_workflows)

