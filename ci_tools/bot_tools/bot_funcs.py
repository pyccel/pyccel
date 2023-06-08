import os
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

tests_with_base = ('coverage', 'doc_coverage', 'pyccel_lint')

comment_folder = os.path.join(os.path.dirname(__file__), '..', 'bot_messages')

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

    def __init__(self, repo, pr_id, commit = None):
        self._repo = repo
        self._GAI = GitHubAPIInteractions(self._repo)
        self._pr_id = pr_id
        self._pr_details = self._GAI.get_pr_details(pr_id)
        if commit:
            self._ref = commit
            self._base = None
        else:
            self._ref = self._pr_details["head"]["sha"]
            self._base = self._pr_details["base"]["sha"]

    def show_tests(self):
        self._GAI.create_comment(self._pr_id, message_from_file('show_tests.txt'))

    def show_commands(self):
        self._GAI.create_comment(self._pr_id, message_from_file('bot_commands.txt'))

    def trust_user(self):
        pass

    def run_tests(self, tests, python_version = None):
        if any(t not in default_python_versions for t in tests):
            self._GAI.create_comment(self._pr_id, "There are unrecognised tests.\n"+message_from_file('show_tests.txt'))
        elif self._pr_details["mergeable_state"] != "clean":
            self._GAI.create_comment(self._pr_id, message_from_file('merge_target.txt'))
        else:
            already_triggered = [c["name"] for c in self._GAI.get_check_runs(self._ref)['check_runs']]
            print(already_triggered)
            for t in tests:
                if any("({t})" in a for a in already_triggered):
                    continue
                pv = python_version or default_python_versions[t]
                inputs = {'python_version':pv, 'ref':self._ref}
                if t in tests_with_base:
                    inputs['base'] = self._base
                self._GAI.run_workflow(f'{t}.yml', inputs)

    def mark_as_draft(self):
        cmds = [github_cli, 'pr', 'ready', str(self._pr_id)]

        with subprocess.Popen(cmds) as p:
            _, err = p.communicate()
        print(err)

    def mark_as_ready(self):
        cmds = [github_cli, 'pr', 'ready', str(self._pr_id), '--undo']

        with subprocess.Popen(cmds) as p:
            _, err = p.communicate()
        print(err)

    def post_coverage_review(self, comments):
        message = message_from_file('coverage_review_message.txt')
        self._GAI.create_review(self._pr_id, self._commit, message, comments)

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
        merged_prs = self._GAI.get_merged_prs()
        has_merged_pr = any(pr for pr in merged_prs if pr['user']['login'] == user)
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
