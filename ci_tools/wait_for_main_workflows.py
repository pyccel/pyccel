import os
from bot_tools.github_api_interactions import GitHubAPIInteractions

GAI = GitHubAPIInteractions()
required_workflows = ('Anaconda-Linux', 'Anaconda-Windows', 'Intel unit tests',
                      'Linux unit tests', 'MacOSX unit tests', 'Windows unit tests',
                      'Pickled-installation', 'Wheel-pickled-installation')
assert GAI.wait_for_runs(os.environ['COMMIT'], required_workflows)

