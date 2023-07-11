import json
import os
from bot_tools.bot_funcs import Bot, test_dependencies

pr_test_keys = ['linux', 'windows', 'macosx', 'coverage', 'docs', 'pylint',
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
    q_name = q['name']
    deps = test_dependencies.get(bot.get_name_key(q_name), ())
    if name_key in deps:
        if all(d in successful_runs for d in deps):
            q_key = q_name.split('(')[1].split(')')[0].strip()
            q_name, python_version = q_key.split(',')
            workflow_ids = None
            if q_name == 'coverage':
                workflow_ids = [int(r['details_url'].split('/')[-1]) for r in runs if r['conclusion'] == "success" and '(' in r['name']]
            bot.run_test(q_name, python_version, q["id"], workflow_ids)
        elif all(d in completed_runs for d in deps):
            bot.GAI.update_run(q["id"], {'conclusion':'cancelled', 'status':"completed"})

draft = bot.is_pr_draft()

if not draft:

    events = bot.GAI.get_events(bot._pr_id)

    print(events)

    shas = [e.get('sha', None) for e in events]
    print(shas)
    print(event['check_run']['head_sha'])
    start_idx = next(s == event['check_run']['head_sha'] for s in shas)
    try:
        end_idx = next(s is not None for s in shas[start_idx+1:])
    except StopIteration:
        end_idx = len(shas)

    relevant_events = events[:end_idx]
    print(start_idx, end_idx)

    print()
    print("---------------------------------------------------------------------------")
    print()

    print(relevant_events)

    event_types = [e['event'] for e in relevant_events]

    ready_events = [e for e in event_types if e in ('ready_for_review', 'convert_to_draft')]

    if ready_events and ready_events[-1] == 'ready_for_review':
        if event['check_run']['conclusion'] not in ('success', 'skipped'):
            bot.mark_as_draft()
        elif all(k in completed_runs for k in pr_test_keys) and \
             all(k in successful_runs for k in pr_test_keys):
            print("TODO")
