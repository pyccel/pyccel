""" Script to find all artifacts that can be passed to the coverage test.
"""
import os
import shutil
import subprocess
import sys
from bot_tools.github_api_interactions import GitHubAPIInteractions

if __name__ == '__main__':
    artifact_urls = sys.argv[1:]
    file = '.coverage'

    GAI = GitHubAPIInteractions()
    for i,url in enumerate(artifact_urls):
        GAI.download_artifact('artifact.zip', url)
        unzip = shutil.which('unzip')
        with subprocess.Popen([unzip, 'artifact.zip']) as p:
            _, err = p.communicate()
        if err:
            print(err)
        os.remove('artifact.zip')
        shutil.move(file, f"{file}.{i}")
