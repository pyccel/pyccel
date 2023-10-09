""" File containing the class Bot and all functions useful for bot reactions.
"""
from datetime import datetime
import json
import os
import platform
import shutil
import subprocess
import time
from .github_api_interactions import GitHubAPIInteractions

default_python_versions = {
        'anaconda_linux': '3.10',
        'anaconda_windows': '3.10',
        'coverage': '3.7',
        'docs': '3.8',
        'linux': '3.7',
        'macosx': '3.10',
        'pickle_wheel': '3.7',
        'pickle': '3.8',
        'editable_pickle': '3.8',
        'pyccel_lint': '3.8',
        'pylint': '3.8',
        'spelling': '3.8',
        'windows': '3.8'
        }

test_names = {
        'anaconda_linux': "Unit tests on Linux with anaconda",
        'anaconda_windows': "Unit tests on Windows with anaconda",
        'coverage': "Coverage verification",
        'docs': "Check documentation",
        'linux': "Unit tests on Linux",
        'macosx': "Unit tests on MacOSX",
        'pickle_wheel': "Test pickling during wheel installation",
        'pickle': "Test pickling during source installation",
        'editable_pickle': "Test pickling during editable source installation",
        'pyccel_lint': "Pyccel best practices",
        'pylint': "Python linting",
        'spelling': "Spelling verification",
        'windows': "Unit tests on Windows"
        }

test_dependencies = {'coverage':['linux']}

tests_with_base = ('coverage', 'docs', 'pyccel_lint')

pr_test_keys = ('linux', 'windows', 'macosx', 'coverage', 'docs', 'pylint',
                'pyccel_lint', 'spelling')

review_stage_labels = ["needs_initial_review", "Ready_for_review", "Ready_to_merge"]

senior_reviewer = ['yguclu', 'EmilyBourne']

trust_givers = ['yguclu', 'EmilyBourne', 'ratnania', 'saidctb', 'bauom']

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
        The name of the file to be read.

    Returns
    -------
    str
        The message to be printed.
    """
    with open(os.path.join(comment_folder, filename), encoding="utf-8") as msg_file:
        comment = msg_file.read()
    return comment

class Bot:
    """
    Class containing all standard bot interactions.

    A class which contains different functionalities for interacting with
    the bot. This class should be used to avoid duplication elsewhere
    (e.g. between the bot triggered on a comment and triggered by marking
    as ready).

    Parameters
    ----------
    pr_id : int, optional
        The number of the PR of interest.

    check_run_id : int, optional
        The id of the current check run (used to update any details).

    commit : str
        The SHA of the current commit.
    """

    def __init__(self, pr_id = None, check_run_id = None, commit = None):
        self._repo = os.environ["GITHUB_REPOSITORY"]
        self._source_repo = None
        self._GAI = GitHubAPIInteractions()
        if pr_id is None:
            self._pr_id = os.environ["PR_ID"]
        else:
            self._pr_id = int(pr_id)
        print("PR ID =", self._pr_id)
        if self._pr_id != 0:
            self._pr_details = self._GAI.get_pr_details(pr_id)
            print(self._pr_details)
            self._base = self._pr_details["base"]["sha"]
            self._source_repo = self._pr_details["base"]["repo"]["full_name"]
        if commit:
            self._ref = commit
            if '/' in self._ref:
                _, _, branch = self._ref.split('/')
                branch_info = self._GAI.get_branch_details(branch)
                self._ref = branch_info['commit']['sha']
        else:
            self._ref = self._pr_details["head"]["sha"]

        if check_run_id:
            self._check_run_id = check_run_id

    def create_in_progress_check_run(self, test):
        """
        Create a check run for key `test` to describe the run in progress.

        Create a new check run which describes the test `test`. Mark the
        run as in progress. This means that the test is run by this
        workflow. The python version used in the key is therefore deduced
        from the current environment.

        Parameters
        ----------
        test : str
            The key for the test. Must be a key in `default_python_versions`
            and `test_names`. It should also be the name of a yml file in
            the folder .github/workflows.

        Returns
        -------
        dict
            A dictionary describing all properties of the new check run.

        Raises
        ------
        AssertionError
            An assertion error is raised if the check run was not successfully posted.
        """
        pv = '.'.join(platform.python_version_tuple()[:2])
        key = f"({test}, {pv})"
        name = f"{test_names[test]} {key}"
        posted = self._GAI.create_run(self._ref, name)
        return posted

    def post_in_progress(self, rerequest = False):
        """
        Update a check run to indicate that the run is in progress.

        Update an existing check run using the id specified at the construction
        to indicate that the run is in progress.

        Parameters
        ----------
        rerequest : bool
            True if the post is due to a test being rerun, False otherwise.

        Returns
        -------
        dict
            A dictionary describing all properties of the check run.

        Raises
        ------
        AssertionError
            An assertion error is raised if the check run was not successfully updated.
        """
        if rerequest and self._check_run_id:
            return self._GAI.rerequest_run(self._check_run_id).json()
        inputs = {
                "status":"in_progress",
                "details_url": f"https://github.com/{self._repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
                }
        return self._GAI.update_run(self._check_run_id, inputs).json()

    def post_completed(self, conclusion):
        """
        Update a check run to indicate that the run is completed.

        Update an existing check run using the id specified at the construction
        to indicate that the run completed with the specified conclusion.

        Parameters
        ----------
        conclusion : str
            The conclusion of the test. Must be one of:
            [action_required, cancelled, failure, neutral, success, skipped, stale, timed_out].

        Raises
        ------
        AssertionError
            An assertion error is raised if the check run was not successfully updated.
        """
        if os.path.exists('test_json_result.json'):
            with open('test_json_result.json', 'r', encoding='utf-8') as f:
                result = json.load(f)
        else:
            result = {}
        print(result)
        params = {
                "status": "completed",
                "conclusion": conclusion,
                }
        if result:
            if "annotations" in result:
                if(len(result["annotations"]) > 50):
                    result["annotations"] = result["annotations"][0:50:1]
            result['title'] = os.environ['GITHUB_WORKFLOW']
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
        """
        Print a comment describing the available tests.

        Print a comment on the current pull request describing the available tests.
        """
        self._GAI.create_comment(self._pr_id, message_from_file('show_tests.txt'))

    def show_commands(self):
        """
        Print a comment describing the available commands.

        Print a comment on the current pull request describing the available bot
        commands.
        """
        self._GAI.create_comment(self._pr_id, message_from_file('bot_commands.txt'))

    def run_tests(self, tests, python_version = None, force_run = False):
        """
        Run the specified tests on the requested python version.

        Run the specified tests on the requested python version. If no version
        is explicitly requested then use the default version as defined in the
        dictionary default_python_versions.

        Parameters
        ----------
        tests : list of str
            A list of keys for the tests. The keys must be in `default_python_versions`
            and `test_names`. They should also be the names of yml files in
            the folder .github/workflows.

        python_version : str, optional
            The requested python version.

        force_run : bool, default=False
            Force the tests to run even if they are not necessary.

        Returns
        -------
        list of str
            A list containing the state of each test.

        See Also
        --------
        Bot.run_test
            Called by this function. It runs individual tests but requires more information.
        """
        if any(t not in default_python_versions for t in tests):
            self._GAI.create_comment(self._pr_id, "There are unrecognised tests.\n"+message_from_file('show_tests.txt'))
            return []
        else:
            check_runs = self._GAI.get_check_runs(self._ref)['check_runs']
            already_triggered = [c["name"] for c in check_runs if c['status'] in ('completed', 'in_progress') and c['conclusion'] != 'cancelled']
            already_triggered_names = [self.get_name_key(t) for t in already_triggered]
            already_programmed = {c["name"]:c for c in check_runs if c['status'] == 'queued'}
            success_names = [self.get_name_key(c["name"]) for c in check_runs if c['status'] == 'completed' and c['conclusion'] == 'success']
            print(already_triggered)
            states = []

            if not force_run:
                # Get a list of all commits on this branch
                cmds = [git, 'log', '--pretty=oneline', '--first-parent', self._ref]
                with subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
                    out, err = p.communicate()
                    print(err)
                    assert p.returncode == 0

                commit_log = [o.split(' ')[0] for o in out.split('\n')]
                print(commit_log)
                idx = next((i for i,c in enumerate(commit_log) if c ==self._base), len(commit_log))
                commit_log = commit_log[:idx+1]

            for t in tests:
                pv = python_version or default_python_versions[t]
                key = f"({t}, {pv})"
                if any(key in a for a in already_triggered):
                    continue
                name = f"{test_names[t]} {key}"
                if not force_run and not self.is_test_required(commit_log, name, t, states):
                    continue
                states.append('queued')
                if key not in already_programmed:
                    posted = self._GAI.prepare_run(self._ref, name)
                else:
                    posted = already_programmed[key]

                deps = test_dependencies.get(t, ())
                print(already_triggered_names, deps)
                if all(d in success_names for d in deps):
                    workflow_ids = None
                    if t == 'coverage':
                        print([r['details_url'] for r in check_runs if r['conclusion'] == "success"])
                        workflow_ids = [int(r['details_url'].split('/')[-1]) for r in check_runs if r['conclusion'] == "success" and '(' in r['name']]
                    print("Running test")
                    self.run_test(t, pv, posted["id"], workflow_ids)
            return states

    def run_test(self, test, python_version, check_run_id, workflow_ids = None):
        """
        Run the specified test on the specified python version.

        Run the specified test on the specified python version by dispatching the necessary
        workflow. The check run id must also be provided to the job so the status can be
        correctly updated from queued. If the job requires any artifacts then workflow_ids
        of the workflows which provided the artifacts must also be provided.

        Parameters
        ----------
        test : str
            The key for the test. Must be a key in `default_python_versions`
            and `test_names`. It should also be the name of a yml file in
            the folder .github/workflows.

        python_version : str
            The requested python version.

        check_run_id : int
            The id of the queued check run.

        workflow_ids : list of int, optional
            The ids of any workflows which may provide the necessary artifacts.
        """
        source_repo = self._source_repo or self._repo
        inputs = {'python_version' : python_version,
                  'ref' : self._ref,
                  'check_run_id' : str(check_run_id),
                  'pr_repo' : source_repo
                 }
        if test in tests_with_base:
            inputs['base'] = self._base
        if test == 'coverage':
            assert workflow_ids is not None
            possible_artifacts = self._GAI.get_artifacts('coverage-artifact')['artifacts']
            print("possible_artifacts : ", possible_artifacts)
            acceptable_urls = [a['archive_download_url'] for a in possible_artifacts if a['workflow_run']['id'] in workflow_ids]
            ntests = 0
            while len(acceptable_urls) == 0 and ntests < 10:
                # Occasionally artifacts are not available immediately after linux concludes
                time.sleep(10)
                possible_artifacts = self._GAI.get_artifacts('coverage-artifact')['artifacts']
                print("possible_artifacts : ", possible_artifacts)
                acceptable_urls = [a['archive_download_url'] for a in possible_artifacts if a['workflow_run']['id'] in workflow_ids]
                ntests += 1
            print("acceptable_urls: ", acceptable_urls)
            inputs['artifact_urls'] = ' '.join(acceptable_urls)
            inputs['pr_id'] = str(self._pr_id)
        elif test == "editable_pickle":
            test = "pickle"
            inputs["editable_string"] = "-e"
        print("Post workflow")
        self._GAI.run_workflow(f'{test}.yml', inputs)

    def is_test_required(self, commit_log, name, key, state):
        """
        Check if a costly test is required.

        Check amongst previous commits. If no Python files have been changed since a
        commit where the check was run then post the result of the previous check.
        Otherwise indicate that the test should be run.

        Parameters
        ----------
        commit_log : list of str
            A list of all commits on this branch.

        name : str
            The name of the test we want to run.

        key : str
            The key which identifies the test.

        state : list of str
            A list to which the state should be appended if found.

        Returns
        -------
        bool
            True if the test should be run, False otherwise.
        """
        print("Checking : ", name)
        if key in ('linux', 'windows', 'macosx', 'anaconda_linux', 'anaconda_windows', 'coverage'):
            has_relevant_change = lambda diff: any((f.startswith('pyccel/') or f.startswith('tests/')) \
                                                    and f.endswith('.py') and f != 'pyccel/version.py' \
                                                    for f in diff) #pylint: disable=unnecessary-lambda-assignment
        elif key in ('pyccel_lint'):
            has_relevant_change = lambda diff: any(f.startswith('pyccel/') and f.endswith('.py') \
                                                    and f != 'pyccel/version.py' for f in diff) #pylint: disable=unnecessary-lambda-assignment
        elif key in ('pylint'):
            has_relevant_change = lambda diff: any(f == 'pyccel/parser/semantic.py' for f in diff) #pylint: disable=unnecessary-lambda-assignment
        elif key in ('docs'):
            has_relevant_change = lambda diff: any(f.endswith('.py') and f != 'pyccel/version.py' \
                                                    for f in diff) #pylint: disable=unnecessary-lambda-assignment
        elif key in ('spelling'):
            has_relevant_change = lambda diff: any(f.endswith('.md') or f == '.dict_custom.txt' for f in diff) #pylint: disable=unnecessary-lambda-assignment
        elif key in ('pickle', 'pickle_wheel', 'editable_pickle'):
            has_relevant_change = lambda diff: any(f.startswith('pyccel/') and f.endswith('.py') \
                                                    and f != 'pyccel/version.py' for f in diff) #pylint: disable=unnecessary-lambda-assignment
        else:
            raise NotImplementedError(f"Please update for new has_relevant_change : {key}")

        for c in commit_log:
            diff = self.get_diff(c)
            if has_relevant_change(diff):
                print("Contains relevant change : ", c)
                return True
            else:
                check_runs = self.get_check_runs(c)
                print(c,':', check_runs)
                try:
                    previous_state = next(cr for cr in check_runs if cr['name'] == name)
                except StopIteration:
                    continue
                conclusion = previous_state['conclusion']
                if conclusion in ('failure', 'success'):
                    if key == 'coverage' and conclusion == 'failure':
                        return True
                    self._GAI.create_run_from_old(self._ref, name, previous_state)
                    state.append(conclusion)
                    return False
        return True

    def mark_as_draft(self):
        """
        Mark the pull request as a draft.

        Mark the pull request specified in the constructor as a draft.
        """
        cmds = [github_cli, 'pr', 'ready', str(self._pr_id), '--undo']

        with subprocess.Popen(cmds) as p:
            _, err = p.communicate()
        print(err)
        self._GAI.clear_labels(self._pr_id, review_stage_labels)

    def draft_due_to_failure(self):
        """
        Mark the pull request as a draft following test failures.

        Mark the pull request specified in the constructor as a draft.
        This function should be called when one of the pull request
        check runs has failed.
        """
        self.mark_as_draft()
        self._GAI.create_comment(self._pr_id, message_from_file('set_draft_failing.txt'))

    def draft_due_to_changes_requested(self, author, reviewer):
        """
        Mark the pull request as a draft following requested changes.

        Mark the pull request specified in the constructor as a draft.
        This function should be called when a review is left on a pull
        request by a user (non-bot) requesting changes.

        Parameters
        ----------
        author : str
            The login id of the author of the pull request.

        reviewer : str
            The login id of the reviewer of the pull request.
        """
        self.mark_as_draft()
        self._GAI.create_comment(self._pr_id, message_from_file('set_draft_changes.txt').format(author=author, reviewer=reviewer))

    def request_mark_as_ready(self):
        """
        Remove the draft status from the pull request.

        Remove the draft status from the pull request specified in the constructor. This
        action is only carried out if the pull request has a description and all items
        on the checklist have been ticked off.

        Returns
        -------
        bool
            Indicates if the pull request could be marked as ready.
        """
        description = self._pr_details['body']
        if description:
            words = description.split()
        else:
            words = []

        if len(words) < 3:
            self._GAI.create_comment(self._pr_id, message_from_file('set_draft_no_description.txt'))
            self.mark_as_draft()
            return False

        welcome_comment = next(c for c in self._GAI.get_comments(self._pr_id) if c['user']['type'] == 'Bot' and c['body'].startswith('Hello'))
        if '- [ ]' in welcome_comment['body']:
            self._GAI.create_comment(self._pr_id, message_from_file('set_draft_checklist_incomplete.txt').format(url = welcome_comment['html_url']))
            self.mark_as_draft()
            return False

        states = self.run_tests(pr_test_keys)

        if 'failure' in states:
            self.draft_due_to_failure()
            return False
        else:
            cmds = [github_cli, 'pr', 'ready', str(self._pr_id)]

            with subprocess.Popen(cmds) as p:
                _, err = p.communicate()
            print(err)

            if all(s == 'success' for s in states):
                self.mark_as_ready(False)

            return True

    def mark_as_ready(self, following_review):
        """
        Mark a pull request as ready for review  by adding the appropriate labels.

        Mark a pull request as ready for review  by adding the appropriate labels.
        The review stage is determined via the function check_review_stage. If
        the stage has changed then a comment is left to indicate who should pay
        attention to the next stage.

        Parameters
        ----------
        following_review : bool
            True if the stage changed following a review, False if it changed
            due to exiting draft status.
        """
        pr_id = self._pr_id
        current_labels = self._GAI.get_current_labels(pr_id)
        print(current_labels)
        stage_labels = [l["name"] for l in current_labels if l["name"] in review_stage_labels]
        assert len(stage_labels) <= 1
        if stage_labels:
            current_stage = stage_labels[0]
        else:
            current_stage = None

        self._GAI.clear_labels(pr_id, stage_labels)
        new_stage, reviews = self.check_review_stage(pr_id)
        self._GAI.add_labels(pr_id, [new_stage])
        author = self._pr_details["user"]["login"]
        approving_reviewers = [reviewer for reviewer, r in reviews.items() if r["state"] == 'APPROVED']
        requested_changes = [reviewer for reviewer, r in reviews.items() if r["state"] == 'CHANGES_REQUESTED']

        try:
            current_stage_index = review_stage_labels.index(current_stage)
        except ValueError:
            current_stage_index = -1
        review_stage_index = review_stage_labels.index(new_stage)

        if following_review:
            if current_stage_index < review_stage_index and new_stage == 'Ready_for_review':
                names = ', '.join(f'@{r}' for r in senior_reviewer)
                approved = ', '.join(f'@{a}' for a in approving_reviewers)
                message = message_from_file('senior_review.txt').format(
                                reviewers=names, author=author, approved=approved)
                self._GAI.create_comment(pr_id, message)
                self._GAI.request_reviewers(pr_id, reviewers=senior_reviewer)
        elif requested_changes:
            requested = ', '.join(f'@{r}' for r in requested_changes)
            message = message_from_file('rerequest_review.txt').format(
                                            reviewers=requested, author=author)
            self._GAI.create_comment(pr_id, message)
            self._GAI.request_reviewers(pr_id, reviewers=requested_changes)
        elif new_stage == 'Ready_for_review':
            names = ', '.join(f'@{r}' for r in senior_reviewer)
            approved = ', '.join(f'@{a}' for a in approving_reviewers)
            message = message_from_file('senior_review.txt').format(
                            reviewers=names, author=author, approved=approved)
            self._GAI.create_comment(pr_id, message)
            self._GAI.request_reviewers(pr_id, reviewers=senior_reviewer)
        elif new_stage == "needs_initial_review":
            message = message_from_file('new_pr.txt').format(author=author)
            self._GAI.create_comment(pr_id, message)
            self._GAI.request_reviewers(pr_id, request_team = True)

    def post_coverage_review(self, comments, approve):
        """
        Create a review describing the coverage results.

        Create a review describing the coverage results. Changes are requested if
        the test failed. The appropriate message is decided here.

        Parameters
        ----------
        comments : list of dict
            A list of dictionaries describing the comments to be left on code snippets.

        approve : bool
            Indicates if the review should approve the pull request or not.
        """
        if approve:
            message = message_from_file('coverage_ok.txt')
            status = 'APPROVE'
        else:
            message = message_from_file('coverage_review_message.txt')
            status = 'REQUEST_CHANGES'
        self._GAI.create_review(self._pr_id, self._ref, message, status, comments)

    def is_user_trusted(self, user):
        """
        Indicate if the user is trusted by the App.

        Detect whether the indicated user is trusted by the App.
        A user is considered trustworthy if they are in the pyccel-dev
        team, if they have previously created a PR which was merged, or
        if a senior dev has declared them trustworthy.

        Parameters
        ----------
        user : str
            The id of the user of interest.

        Returns
        -------
        bool
            True if the user is trusted, false otherwise.
        """
        print("Trusted?")
        in_team = self._GAI.check_for_user_in_team(user, 'pyccel-dev')
        if in_team["message"] != "Not found":
            print("In team")
            return True
        print("User not in team")
        merged_prs = self._GAI.get_prs('all')
        print(merged_prs, user)
        has_merged_pr = any(pr for pr in merged_prs if pr['user']['login'] == user and pr['merged_at'])
        if has_merged_pr:
            print("Merged PR")
            return has_merged_pr
        print("User has no merged PRs")
        comments = self._GAI.get_comments(self._pr_id)
        comments_from_trust_givers = [c['body'].split() for c in comments if c['user']['login'] in trust_givers]
        expected_trust_command = ['/bot', 'trust', 'user', user]
        awarded_trust = any(c[:4] == expected_trust_command for c in comments_from_trust_givers)
        return awarded_trust

    def warn_untrusted(self):
        """
        Print a comment indicating that the user is not trusted.

        Print a comment on the current pull request warning the user that they are not
        yet trusted.
        """
        self._GAI.create_comment(self._pr_id, message_from_file('untrusted_user.txt'))

    def indicate_trust(self, user):
        """
        Leave a message explaining that a user was recognised as trustworthy.

        Leave a message on the current pull request indicating that a user was recognised
        as trustworthy and can now run tests.

        Parameters
        ----------
        user : str
            The id of the user of interest.
        """
        if user.startswith('@'):
            user = user[1:]
        self._GAI.create_comment(self._pr_id, message_from_file('trusting_user.txt').format(user=user))

    def check_review_stage(self, pr_id):
        """
        Find the review stage.

        Use the GitHub CLI to examine the reviews left on the pull request
        and determine the current stage of the review process. If a senior
        reviewer has approved then the stage is "Ready_to_merge". Otherwise
        if everyone else who has left a review (excluding a bot) has approved
        then the stage is "Ready_for_review". If not then the stage is simply
        "needs_initial_review".

        Parameters
        ----------
        pr_id : int
            The number of the PR.

        Returns
        -------
        str
            The review stage.

        dict
            A dictionary whose keys are users (non-bots) who left reviews and
            whose values are dictionaries describing the reviews which either
            approved or requested changes.
        """
        all_reviews = [r for r in self._GAI.get_reviews(self._pr_id) if r['user']['type'] != 'Bot' and r['state'] in ('APPROVED', 'CHANGES_REQUESTED')]
        all_reviews.sort(key=lambda r: datetime.fromisoformat(r['submitted_at'].strip('Z')))
        reviews = {r['user']['login'] : r for r in all_reviews}
        if any(reviewer in senior_reviewer and r["state"] == 'APPROVED' for reviewer, r in reviews.items()):
            return "Ready_to_merge", reviews

        non_senior_reviews = [r for reviewer, r in reviews.items() if reviewer not in senior_reviewer]

        if non_senior_reviews and all(r["state"] == 'APPROVED' for r in non_senior_reviews):
            return "Ready_for_review", reviews

        else:
            return "needs_initial_review", reviews

    def get_check_runs(self, commit = None):
        """
        Get a list of all check runs which have run on this commit.

        Get a dictionary containing all information about check runs which have run
        on the commit specified at the constructor.

        Parameters
        ----------
        commit : str, optional
            The commit for which we wish to get check run information.
            The default value is the most recent commit associated with this
            pull request.

        Returns
        -------
        dict
            A dictionary describing the check runs.
        """
        if commit is None:
            commit = self._ref
        result = self._GAI.get_check_runs(commit)
        print("get_check_runs")
        print(result)
        return result['check_runs']

    def get_bot_review_comments(self):
        """
        Get all review comments related to a review left by the bot.

        Get all review comment threads on code snippets for reviews left by the bot.
        Any outdated comments are discarded and the author is congratulated for fixing
        their coverage error.

        Returns
        -------
        list of list of dict
            A list of comments on a code snippet. Each element of the list is a list of
            comments left on one particular snippet.
        """
        comments = self._GAI.get_review_comments(self._pr_id)
        grouped_comments = {}

        for c in comments:
            c_id = c.get('in_reply_to_id', c['id'])
            grouped_comments.setdefault(c_id, []).append(c)

        bot_grouped_comments =[c for c in grouped_comments.values() if c[0]['user'].get('type', 'user') == 'Bot']

        relevant_comments = [c for c in bot_grouped_comments if c[0]['position'] is not None]
        discarded_comments = [c for c in bot_grouped_comments if c[0]['position'] is None]

        for comment_thread in discarded_comments:
            c = comment_thread[0]
            self.accept_coverage_fix(comment_thread)

        return relevant_comments

    def get_pr_id(self):
        """
        Get the id of the current pull request.

        Get the id of the current pull request. Where this was provided
        in the constructor it is returned, otherwise pull requests are
        examined until one is found whose head SHA matches the commit
        examined here.

        Returns
        -------
        int
            The id of the current pull request.
        """
        if self._pr_id:
            return self._pr_id
        else:
            cmds = [git, 'branch', '-a', '--contains', self._ref]
            with subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
                out, err = p.communicate()
                print(err)
                assert p.returncode == 0
            branches = out.split('\n')
            if len(branches) == 1:
                branch = branches[0].split('/')[-1]
                cmds = [github_cli, 'pr', 'list', '--head', branch, '--json', 'number']
                with subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as p:
                    out, err = p.communicate()
                    print(err)
                    assert p.returncode == 0
                self._pr_id = json.loads(out)[0]['number']
            else:
                possible_prs = self._GAI.get_prs()
                print(possible_prs)
                self._pr_id = next(pr['number'] for pr in possible_prs if pr['head']['sha'] == self._ref)
            self._pr_details = self._GAI.get_pr_details(self._pr_id)
            self._base = self._pr_details["base"]["sha"]
            self._source_repo = self._pr_details["base"]["repo"]["full_name"]
            return self._pr_id

    def get_diff(self, base_commit = None):
        """
        Get the diff between the base and the current commit.

        Get the git description of the difference between the current
        commit and the specified base commit. This output
        shows how github organises the files tab and allows line
        numbers do be calculated from git blob positions.

        Parameters
        ----------
        base_commit : str, optional
            The commit against which the current commit should be compared.
            The default value is the base commit of the pull request.

        Returns
        -------
        dict
            A dictionary whose keys are files and whose values are lists of
            lines which appear in the diff including code and blob headers.
        """
        if base_commit is None:
            base_commit = self._base
        assert bool(base_commit)
        cmd = [git, 'diff', f"{base_commit}..{self._ref}"]
        print(cmd)
        with subprocess.Popen(cmd + ['--name-only'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as p:
            out, _ = p.communicate()
        diff = {f: None for f in out.strip().split('\n')}
        for f in diff:
            with subprocess.Popen(cmd + [f], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as p:
                out, err = p.communicate()
            if not err:
                lines = out.split('\n')
                n = next((i for i,l in enumerate(lines) if '@@' in l), len(lines))
                diff[f] = lines[n:]
        return {f:l for f,l in diff.items() if l is not None}

    def accept_coverage_fix(self, comment_thread):
        """
        Leave a message on the pull request indicating that the coverage was fixed.

        Leave a message on the pull request specified in the constructor, indicating
        that the coverage problem designated by the comment in the specified thread,
        was fixed.

        Parameters
        ----------
        comment_thread : list of dict
            A list of dictionaries describing comments on the same review thread.
        """
        message = message_from_file('accept_coverage_fix.txt')
        print(comment_thread)
        if any(c['body'] == message for c in comment_thread):
            return
        target = comment_thread[0]
        comment_id = target.get('in_reply_to_id', target['id'])
        print("id: ", comment_id)
        reply = self._GAI.create_comment(self._pr_id, message,
                                 reply_to = comment_id)
        print(reply.text)

    def is_pr_draft(self):
        """
        Indicate whether the pull request is a draft.

        Indicate whether the pull request is a draft.

        Returns
        -------
        bool
            True if draft, False otherwise.
        """
        return self._pr_details['draft']

    def leave_comment(self, comment):
        """
        Leave a comment on the pull request.

        Leave the specified comment on the pull request.

        Parameters
        ----------
        comment : str
            The comment to be left on the pull request.
        """
        self._GAI.create_comment(self._pr_id, comment)

    @property
    def GAI(self):
        """
        Get the GitHubAPIInteractions object.

        Get the GitHubAPIInteractions object to allow for direct
        manipulation of the API.
        """
        return self._GAI

    @property
    def repo(self):
        """
        Get the full name of the repository being handled.

        Get the full name of the repository being handled.
        """
        return self._repo

    def get_name_key(self, name):
        """
        Get the name used as a key from the full run name.

        Get the name used as a key to dictionaries including test_names and
        default_python_versions from the full name reported in the check
        run.

        Parameters
        ----------
        name : str
            The name saved in the check run.

        Returns
        -------
        str
            The name which can be used as a key.
        """
        if '(' in name:
            return name.split('(')[1].split(',')[0]
        elif 'Codacy' in name:
            return 'Codacy'
        else:
            return name
