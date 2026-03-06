""" File containing the class Bot and all functions useful for bot reactions.
"""
import json
import os
import shutil
import subprocess
from .github_api_interactions import GitHubAPIInteractions

# Tests which require the base branch to be passed as an argument to the workflow dispatch
# These tests only check the state of new code

comment_folder = os.path.join(os.path.dirname(__file__), '..', 'bot_messages')

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


    commit : str
        The SHA of the current commit.
    """

    def __init__(self, pr_id = None, commit = None):
        self._repo = os.environ["GITHUB_REPOSITORY"]
        self._source_repo = None
        if pr_id is None:
            self._pr_id = os.environ["PR_ID"]
        else:
            self._pr_id = int(pr_id)
        print("PR ID =", self._pr_id)
        if self._pr_id != 0:
            GAI = GitHubAPIInteractions(self._repo)
            self._pr_details = GAI.get_pr_details(pr_id)
            self._base = self._pr_details["base"]["sha"]
            self._source_repo = self._pr_details["base"]["repo"]["full_name"]
        self._GAI = GitHubAPIInteractions(self._source_repo)
        if commit:
            self._ref = commit
            if '/' in self._ref:
                _, _, branch = self._ref.split('/',2)
                branch_info = self._GAI.get_branch_details(branch)
                self._ref = branch_info['commit']['sha']
        else:
            self._ref = self._pr_details["head"]["sha"]

    def post_completed(self, name, conclusion):
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
            self._GAI.post_coverage_run(self._ref, name, params)
        except AssertionError as a:
            params = {
                    "status": "completed",
                    "conclusion": "failure",
                    }
            self._GAI.post_coverage_run(self._ref, name, params)
            raise a

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
        print(' '.join(cmd))
        with subprocess.Popen(cmd + ['--name-only'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as p:
            out, _ = p.communicate()
        diff = {f: None for f in out.strip().split('\n')}
        for f in diff:
            with subprocess.Popen(cmd + ['--', f], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) as p:
                out, err = p.communicate()
            if not err:
                lines = out.split('\n')
                n = next((i for i,l in enumerate(lines) if '@@' in l), len(lines))
                diff[f] = lines[n:]
            else:
                print(err)
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
