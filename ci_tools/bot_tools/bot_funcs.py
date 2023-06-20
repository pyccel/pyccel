import json
import os
import platform
import shutil
import subprocess
from .github_api_interactions import GitHubAPIInteractions

default_python_versions = {
        'anaconda_linux': '3.10',
        'anaconda_windows': '3.10',
        'coverage': '3.7',
        'doc_coverage': '3.8',
        'linux': '3.7',
        'macosx': '3.10',
        'pickle_wheel': '3.7',
        'pickle': '3.8',
        'pyccel_lint': '3.8',
        'pylint': '3.8',
        'spelling': '3.8',
        'windows': '3.8'
        }

test_names = {
        'anaconda_linux': "Unit tests on linux with anaconda",
        'anaconda_windows': "Unit tests on windowd with anaconda",
        'coverage': "Coverage verification",
        'doc_coverage': "Check documentation",
        'linux': "Unit tests on linux",
        'macosx': "Unit tests on macosx",
        'pickle_wheel': "Test pickling during wheel installation",
        'pickle': "Test pickling during source installation",
        'pyccel_lint': "Pyccel best practices",
        'pylint': "Python linting",
        'spelling': "Spelling verification",
        'windows': "Unit tests on windows"
        }

tests_with_base = ('coverage', 'doc_coverage', 'pyccel_lint')

comment_folder = os.path.join(os.path.dirname(__file__), '..', 'bot_messages')

github_cli = shutil.which('gh')
git = shutil.which('git')

def message_from_file(filename):
    """
    Get the message saved in the file.

    Reads the contents of the file `filename`, located in the
    folder ./bot_messages/. The extracted string is returned for
    use as a comment on a PR.

    Parameters
    ----------
    filename : str
        The name of the file to be read

    Results
    -------
    str : The message to be printed.
    """
    with open(os.path.join(comment_folder, filename), encoding="utf-8") as msg_file:
        comment = msg_file.read()
    return comment

class Bot:
    trust_givers = ['yguclu', 'EmilyBourne', 'ratnania', 'saidctb', 'bauom']

    def __init__(self, pr_id = None, check_run_id = None, commit = None):
        self._repo = os.environ["GITHUB_REPOSITORY"]
        self._GAI = GitHubAPIInteractions()
        if pr_id is None:
            self._pr_id = os.environ["PR_ID"]
        else:
            self._pr_id = pr_id
        if self._pr_id != 0:
            self._pr_details = self._GAI.get_pr_details(pr_id)
            self._base = self._pr_details["base"]["sha"]
        if commit:
            self._ref = commit
        else:
            self._ref = self._pr_details["head"]["sha"]

        if check_run_id:
            self._check_run_id = check_run_id

    def create_in_progress_check_run(self, test):
        pv = '.'.join(platform.python_version_tuple()[:2])
        key = f"({test}, {pv})"
        name = f"{test_names[test]} {key}"
        posted = self._GAI.create_run(self._ref, name)
        return posted

    def post_in_progress(self):
        inputs = {
                "status":"in_progress",
                "details_url": f"https://github.com/{self._repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
                }
        print(inputs)
        return self._GAI.update_run(self._check_run_id, inputs).json()

    def post_completed(self, conclusion):
        if os.path.exists('test_json_result.json'):
            with open('test_json_result.json', 'r') as f:
                result = json.load(f)
        else:
            result = {}
        print(result)
        params = {
                "status": "completed",
                "conclusion": conclusion,
                }
        if result:
            params["output"] = result
        try:
            self._GAI.update_run(self._check_run_id, params)
        except AssertionError as a:
            params = {
                    "status": "completed",
                    "conclusion": "failure",
                    }
            self._GAI.update_run(self._check_run_id, params)
            raise a

    def show_tests(self):
        self._GAI.create_comment(self._pr_id, message_from_file('show_tests.txt'))

    def show_commands(self):
        self._GAI.create_comment(self._pr_id, message_from_file('bot_commands.txt'))

    def trust_user(self):
        pass

    def run_tests(self, tests, python_version = None):
        if any(t not in default_python_versions for t in tests):
            self._GAI.create_comment(self._pr_id, "There are unrecognised tests.\n"+message_from_file('show_tests.txt'))
        else:
            already_triggered = [c["name"] for c in self._GAI.get_check_runs(self._ref)['check_runs']]
            print(already_triggered)
            for t in tests:
                pv = python_version or default_python_versions[t]
                key = f"({t}, {pv})"
                if any(key in a for a in already_triggered):
                    continue
                name = f"{test_names[t]} {key}"
                posted = self._GAI.prepare_run(self._ref, name)
                if t != "coverage":
                    self.run_test(t, pv, posted["id"])

    def run_test(self, test, python_version, check_run_id, workflow_ids = None):
        inputs = {'python_version': python_version, 'ref': self._ref, 'check_run_id': str(check_run_id)}
        if test in tests_with_base:
            inputs['base'] = self._base
        if test == 'coverage':
            assert workflow_ids is not None
            possible_artifacts = self._GAI.get_artifacts('coverage-artifact')['artifacts']
            print("possible_artifacts : ", possible_artifacts)
            acceptable_urls = [a['archive_download_url'] for a in possible_artifacts if a['workflow_run']['id'] in workflow_ids]
            print("acceptable_urls: ", acceptable_urls)
            inputs['artifact_urls'] = ' '.join(acceptable_urls)
            inputs['pr_id'] = str(self._pr_id)
        self._GAI.run_workflow(f'{test}.yml', inputs)

    def mark_as_draft(self):
        cmds = [github_cli, 'pr', 'ready', str(self._pr_id)]

        with subprocess.Popen(cmds) as p:
            _, err = p.communicate()
        print(err)

    def request_mark_as_ready(self):
        cmds = [github_cli, 'pr', 'ready', str(self._pr_id), '--undo']

        with subprocess.Popen(cmds) as p:
            _, err = p.communicate()
        print(err)

    def mark_as_ready(self):
        pass

    def post_coverage_review(self, comments, approve):
        if approve:
            message = message_from_file('coverage_ok.txt')
            status = 'APPROVE'
        else:
            message = message_from_file('coverage_review_message.txt')
            status = 'REQUEST_CHANGES'
        self._GAI.create_review(self._pr_id, self._ref, message, status, comments)

    def is_user_trusted(self, user):
        """
        Is the indicated user is trusted by the App.

        Detect whether the indicated user is trusted by the App.
        A user is considered trustworthy if they are in the pyccel-dev
        team, if they have previously created a PR which was merged, or
        if a senior dev has declared them trustworthy.
        """
        in_team = self._GAI.check_for_user_in_team(user, 'pyccel-dev')
        if in_team["message"] != "Not found":
            return True
        merged_prs = self._GAI.get_prs('all')
        has_merged_pr = any(pr for pr in merged_prs if pr['user']['login'] == user and pr['merged_at'])
        if has_merged_pr:
            return has_merged_pr
        comments = self._GAI.get_comments(self._pr_id)
        comments_from_trust_givers = [c['body'].split() for c in comments if c['user']['login'] in self.trust_givers]
        expected_trust_command = ['/bot', 'trust', 'user', user]
        awarded_trust = any(c[:4] == expected_trust_command for c in comments_from_trust_givers)
        return awarded_trust

    def warn_untrusted(self):
        self._GAI.create_comment(self._pr_id, message_from_file('untrusted_user.txt'))

    def indicate_trust(self, user):
        if user.startswith('@'):
            user = user[1:]
        self._GAI.create_comment(self._pr_id, message_from_file('trusting_user.txt').format(user=user))

    def check_review_stage(self, pr_id):
        """
        Find the review stage.

        Use the GitHub CLI to examine the reviews left on the pull request
        and determine the current stage of the review process.

        Parameters
        ----------
        pr_id : int
            The number of the PR.

        Results
        -------
        bool : Indicates if the PR is ready to merge.

        bool : Assuming the PR is not ready to merge, indicates if the PR is
                ready for a review from a senior reviewer.

        requested_changes : List of authors who requested changes.

        reviews : Summary of all reviews left on the PR.
        """
        reviews, _ = get_review_status(pr_id)
        senior_review = [r for a,r in reviews.items() if a in senior_reviewer]

        other_review = [r for a,r in reviews.items() if a not in senior_reviewer]

        ready_to_merge = any(r.state == 'APPROVED' for r in senior_review) and not any(r.state == 'CHANGES_REQUESTED' for r in senior_review)

        ready_for_senior_review = any(r.state == 'APPROVED' for r in other_review) and not any(r.state == 'CHANGES_REQUESTED' for r in other_review)

        requested_changes = [a for a,r in reviews.items() if r.state == 'CHANGES_REQUESTED']

        return ready_to_merge, ready_for_senior_review, requested_changes, reviews

    def get_check_runs(self):
        return self._GAI.get_check_runs(self._ref)['check_runs']

    def get_bot_review_comments(self):
        all_reviews = self._GAI.get_reviews(self._pr_id)
        comments = [c for r in all_reviews for c in self._GAI.get_review_comments(self._pr_id, r["id"])]

        grouped_comments = {}

        for c in comments:
            c_id = c.get('in_reply_to_id', c['id'])
            grouped_comments.setdefault(c_id, []).append(c)

        relevant_comments = [c for c in grouped_comments if c[0]['position'] is not None]
        discarded_comments = [c for c in grouped_comments if c[0]['position'] is None]

        for comment_thread in discarded_comments:
            c = comment_thread[0]
            self.accept_coverage_fix(comment_thread)

        return relevant_comments

    def get_pr_id(self):
        possible_prs = self._GAI.get_prs()
        self._pr_id = next(pr['number'] for pr in possible_prs if pr['head']['sha'] == self._ref)
        return self._pr_id

    def get_diff(self):
        cmd = [git, 'diff', f"{self._base}..{self._ref}"]
        print(cmd)
        with subprocess.Popen(cmd + ['--name-only'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as p:
            out, _ = p.communicate()
        diff = {f: None for f in out.strip().split('\n')}
        for f in diff:
            with subprocess.Popen(cmd + [f], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as p:
                out, err = p.communicate()
            if not err:
                lines = out.split('\n')
                n = next(i for i,l in enumerate(lines) if '@@' in l)
                diff[f] = lines[n:]
        return {f:l for f,l in diff.items() if l is not None}

    def get_detailed_comments(self, comment_id):
        return self._GAI.get_detailed_comments(comment_id)

    def accept_coverage_fix(self, comment_thread):
        message = message_from_file('accept_coverage_fix.txt')
        if any(c['body'] == message for c in comment_thread):
            return
        target = comment_thread[0]
        comment_id = target.get('in_reply_to_id', target['id'])
        reply = self._GAI.create_comment(self._pr_id, message,
                                 reply_to = comment_id)
        print(reply.text)

    @property
    def GAI(self):
        return self._GAI

    @property
    def repo(self):
        return self._repo
