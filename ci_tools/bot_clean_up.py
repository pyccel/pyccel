import json
import os
from bot_tools.bot_funcs import Bot, test_dependencies

pr_test_keys = ['linux', 'windows', 'macosx', 'coverage', 'doc_coverage', 'pylint',
                'pyccel_lint', 'spelling', 'Codacy']


# Parse event payload from $GITHUB_EVENT_PATH variable
# (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
# The contents of this json file depend on the triggering event and are
# described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
    event = json.load(event_file)

name = event['check_run']['name']

print(event)

print(event['check_run']['pull_requests'])
bot = Bot(pr_id = next(p['number'] for p in event['check_run']['pull_requests']), commit = event['check_run']['head_sha'])

name_key = bot.get_name_key(name)

runs = bot.get_check_runs()

print("Runs: ", runs)

all_run_names = [bot.get_name_key(r['name']) for r in runs]
successful_runs = [n for n,r in zip(all_run_names, runs) if r['conclusion'] == "success"]
completed_runs = [n for n,r in zip(all_run_names, runs) if r['status'] == "completed"]
queued_runs = [r for r in runs if r['status'] == "queued"]

print("Successful:", successful_runs)
print("Completed:", completed_runs)

for q in queued_runs:
    deps = test_dependencies.get(bot.get_name_key(q), ())
    if name_key in deps:
        if all(r in deps for r in successful_runs):
            q_key = q.split('(')[1].split(')')[0].strip()
            q_name, python_version = q_key.split(',')
            workflow_ids = None
            if q_key == 'coverage':
                workflow_ids = [int(r['details_url'].split('/')[-1]) for r in runs if r['conclusion'] == "success"]
            bot.run_test(q_key, python_version, q["id"], workflow_ids)
        elif all(r in deps for r in completed_runs):
            bot.GAI.update_run(q["id"], {'conclusion':'cancelled', 'status':"completed"})

if all(k in completed_runs for k in pr_test_keys):
    if all(k in successful_runs for k in pr_test_keys):
        bot.mark_as_ready()
    else:
        bot.mark_as_draft()
