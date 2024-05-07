import os
from bot_tools.github_api_interactions import GitHubAPIInteractions

GAI = GitHubAPIInteractions()
required_workflows = ('Pickled-installation', 'Wheel-pickled-installation')
assert GAI.wait_for_runs(os.environ['COMMIT'], required_workflows)

