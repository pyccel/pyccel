
coverage_deps = ['linux']

pr_test_keys = ['linux', 'windows', 'macosx', 'coverage', 'doc_coverage', 'pylint',
                'pyccel_lint', 'spelling', 'Codacy']


def get_name_key(name):
    return name.split('(')[1].split(',')[0]

# Parse event payload from $GITHUB_EVENT_PATH variable
# (documented here : https://docs.github.com/en/actions/learn-github-actions/variables#default-environment-variables)
# The contents of this json file depend on the triggering event and are
# described here :  https://docs.github.com/en/webhooks-and-events/webhooks/webhook-events-and-payloads
with open(os.environ["GITHUB_EVENT_PATH"], encoding="utf-8") as event_file:
    event = json.load(event_file)

name = event['check_run']['name']

name_key = get_name_key(name)

runs = GAI.get_check_runs(self._ref)['check_runs']

successful_runs = [get_name_key(r['name']) for r in runs if r['conclusion'] == "success"]
completed_runs = [get_name_key(r['name']) for r in runs if r['status'] == "completed"]

if name_key in coverage_deps:
    coverage_run = next(r for r in runs if get_name_key(r['name']) == 'coverage')
    if all(c in successful_runs for c in coverage_deps):
        python_version = coverage_run["name"].split('(')[1].split(',')[1].split(')')[0].strip()
        inputs = {'ref':event['check_run']['head_sha'], python
        GAI.run_workflow(filename, inputs)
    elif all(c in completed_runs for c in coverage_deps):
        GAI.update_run(coverage_run["id"], {'conclusion':'cancelled', 'status':"completed"})
