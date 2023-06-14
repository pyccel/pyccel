import shutil
import sys
from bot_tools.github_api_interactions import GitHubAPIInteractions

artifact_urls = sys.argv[1:]

GAI = GitHubAPIInteractions()
for i,url in enumerate(artifact_urls):
    GAI.download_artifact(url)
    shutil.move(file, f"{file}.{i}")
