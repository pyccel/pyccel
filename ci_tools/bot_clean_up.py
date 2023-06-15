import json
import os
from bot_tools.bot_funcs import Bot

coverage_deps = ['linux']

pr_test_keys = ['linux', 'windows', 'macosx', 'coverage', 'doc_coverage', 'pylint',
                'pyccel_lint', 'spelling', 'Codacy']


def get_name_key(name):
    if name == "Codacy Static Analysis":
        return "Codacy"
    elif '(' in name:
        return name.split('(')[1].split(',')[0]
    else:
        return name

# Parse event payload from $GITHUB_EVENT_PATH variable
# (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
# The contents of this json file depend on the triggering event and are
# described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
    event = json.load(event_file)

name = event['check_run']['name']

name_key = get_name_key(name)

print(event)

print(event['check_run']['pull_requests'])
bot = Bot(pr_id = next(p['number'] for p in event['check_run']['pull_requests']), commit = event['check_run']['head_sha'])

runs = bot.get_check_runs()

print("Runs: ", runs)

successful_runs = [get_name_key(r['name']) for r in runs if r['conclusion'] == "success"]
completed_runs = [get_name_key(r['name']) for r in runs if r['status'] == "completed"]

print("Successful:", successful_runs)
print("Completed:", completed_runs)

if name_key in coverage_deps:
    coverage_run = next(r for r in runs if get_name_key(r['name']) == 'coverage')
    if all(c in successful_runs for c in coverage_deps):
        python_version = coverage_run["name"].split('(')[1].split(',')[1].split(')')[0].strip()
        workflow_ids = [int(r['details_url'].split('/')[-1]) for r in runs if r['conclusion'] == "success"]
        print("Searching for ids: ", workflow_ids)
        bot.run_test('coverage', python_version, coverage_run["id"], workflow_ids)
    elif all(c in completed_runs for c in coverage_deps):
        bot.GAI.update_run(coverage_run["id"], {'conclusion':'cancelled', 'status':"completed"})

if all(k in completed_runs for k in pr_test_keys):
    if all(k in successful_runs for k in pr_test_keys):
        bot.mark_as_ready()
    else:
        bot.mark_as_draft()
