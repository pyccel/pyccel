import jwt
import os
import time
import requests

def get_authorization():
    """
    Get the token necessary to authentificate as the bot.

    Use the private token of the bot (saved in an environment
    secret) to request a JSON Web Token (JWT). Save that JWT
    and its expiry date to the environment for future actions
    ($GITHUB_ENV).

    Returns
    -------
    str
        The JWT used for authentificating as the bot.
    str
        A string describing the expiration of the JWT.
    """
    signing_key = jwt.jwk_from_pem(bytes(os.environ["PEM"], "utf-8"))
    # Issued at time
    # JWT expiration time (10 minutes maximum)
    # GitHub App's identifier
    payload = {'iat': int(time.time()), 'exp': int(time.time()) + 60, 'iss': 337566}

    jw_token=jwt.JWT().encode(payload, signing_key, alg='RS256')

    headers = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {jw_token}", "X-GitHub-Api-Version": "2022-11-28"}

    # Create JWT
    reply = requests.post("https://api.github.com/app/installations/37820767/access_tokens", headers=headers)

    token  = reply.json()["token"]
    expiry = reply.json()["expires_at"]

    with open(os.environ["GITHUB_ENV"], "r") as f:
        output = f.read()

    if "installation_token" in output:
        lines = output.split('\n')
        print("Parsed : ", lines)
        output = '\n'.join(l for l in lines if "installation_token" not in l)

    with open(os.environ["GITHUB_ENV"], "w") as f:
        f.write(output)
        print(f"installation_token={token}", file=f)
        print(f"installation_token_exp={expiry}", file=f)

    return token, expiry

class GitHubAPIInteractions:
    """
    Class which handles all interactions with the GitHub API.

    A helper class which exposes the GitHub API in a readable
    manner.
    """
    def __init__(self):
        repo = os.environ["GITHUB_REPOSITORY"]
        self._org, self._repo = repo.split('/')
        if "installation_token" in os.environ:
            self._install_token = os.environ["installation_token"]
            self._install_token_exp = time.strptime(os.environ["installation_token_exp"], "%Y-%m-%dT%H:%M:%SZ")
        else:
            self._install_token, expiry = get_authorization()
            self._install_token_exp = time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ")

    def _post_request(self, method, url, json=None, **kwargs):
        """
        Post the request to GitHub.

        Use the requests library to sent the request to the API.

        Parameters
        ----------
        method : str
            The type of request, e.g. GET/POST/PATCH.

        url : str
            The url where the request should be sent.

        json : dictionary, optional
            Any additional information provided with the request.

        **kwargs : dictionary
            Any additional arguments for the requests library.

        Returns
        -------
        requests.Response
            The response collected from the request.
        """
        reply = requests.request(method, url, json=json, headers=self.get_headers(), **kwargs)
        return reply

    def get_branch_details(self, branch_name):
        """
        Get the details of the specified branch.

        Use the GitHub API to get information about the mentioned branch.

        Parameters
        ----------
        branch_name : str
            The name of the branch.

        Returns
        -------
        dict
            A dictionary containing information about the branch.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/branches/{branch_name}"
        return self._post_request("GET", url).json()

    def check_runs(self, commit):
        """
        Get a list of all check runs which were run on the commit.

        Get a list of all check runs which were run in the repository for
        the commit passed as an argument.

        Parameters
        ----------
        commit : str
            The SHA of the commit of interest.

        Returns
        -------
        dict
            A dictionary containing information about the check runs.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/commits/{commit}/check-runs"
        return self._post_request("GET", url).json()

    def create_suite(self, commit):
        """
        Try to create a new check suite which is not rerequestable.

        Create a new check suite on the specified commit and make it non-rerequestable.
        
        Parameters
        ----------
        commit : str
            The commit to be tested.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/check-suites"
        configs = {"head_sha": commit,
                   "rerequestable": False}
        run = self._post_request("POST", url, json)

        print(run.text)

    def create_run(self, commit, name):
        """
        Create a new check run.

        Create a new check run with the specified name which tests the mentioned commit.
        The check run is marked as in progress. The details url is pointed at the
        run summary page for this run.

        Parameters
        ----------
        commit : str
            The commit to be tested.

        name : str
            The name of the check run.

        Returns
        -------
        dict
            A dictionary describing all properties of the new check run.

        Raises
        ------
        AssertionError
            An assertion error is raised if the check run was not successfully posted.
        """
        self.create_suite(commit)
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/check-runs"
        workflow_url = f"https://github.com/{self._org}/{self._repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        print("create_run:", url)
        json = {"name": name,
                "head_sha": commit,
                "status": "in_progress",
                "details_url": workflow_url}
        run = self._post_request("POST", url, json)
        assert run.status_code == 201
        return run.json()

    def prepare_run(self, commit, name):
        """
        Add a new check run to the queue.

        Create a new check run with the specified name which tests the mentioned commit.
        The check run is marked as queued.

        Parameters
        ----------
        commit : str
            The commit to be tested.

        name : str
            The name of the check run.

        Returns
        -------
        dict
            A dictionary describing all properties of the new check run.

        Raises
        ------
        AssertionError
            An assertion error is raised if the check run was not successfully posted.
        """
        self.create_suite(commit)
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/check-runs"
        json = {"name": name,
                "head_sha": commit,
                "status": "queued"}
        run = self._post_request("POST", url, json)
        assert run.status_code == 201
        return run.json()

    def update_run(self, run_id, json):
        """
        Update an existing check run.

        Update information on the check run with id "run_id" using the information
        in the json dictionary as described here:
        https://docs.github.com/en/rest/checks/runs?apiVersion=2022-11-28#update-a-check-run

        Parameters
        ----------
        run_id : int
            The id of the check run.

        json : dictionary
            The information that should be updated in the check run.

        Returns
        -------
        requests.Response
            The response collected from the request.

        Raises
        ------
        AssertionError
            An assertion error is raised if the check run was not successfully updated.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/check-runs/{run_id}"
        run = self._post_request("PATCH", url, json)
        print(run.text)
        assert run.status_code == 200
        return run

    def rerequest_run(self, run_id):
        """
        Rerequest an existing check run.

        Rerequest the check run with id "run_id" as described here:
        https://docs.github.com/en/rest/checks/runs?apiVersion=2022-11-28#rerequest-a-check-run

        Parameters
        ----------
        run_id : int
            The id of the check run.

        Returns
        -------
        requests.Response
            The response collected from the request.

        Raises
        ------
        AssertionError
            An assertion error is raised if the check run was not successfully rerequested.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/actions/runs/{run_id}/rerun"
        run = self._post_request("PATCH", url)
        print(run.text)
        assert run.status_code == 201
        return run

    def get_pr_details(self, pr_id):
        """
        Get the details of a pull request.

        Get all details provided by the API for a given pull request.

        Parameters
        ----------
        pr_id : int
            The id of the pull request.

        Returns
        -------
        dict
            A dictionary describing the properties of the pull request.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/pulls/{pr_id}"
        return self._post_request("GET", url).json()

    def run_workflow(self, filename, inputs):
        """
        Create a workflow dispatch event.

        Create a workflow dispatch event as described in the API docs here:
        https://docs.github.com/en/rest/actions/workflows?apiVersion=2022-11-28

        All workflows are run from the devel branch, they then use the inputs
        to checkout the relevant code.

        Parameters
        ----------
        filename : str
            The name of the file containing the workflow we wish to run.

        inputs : dict
            A dictionary of any inputs required for the workflow.

        Raises
        ------
        AssertionError
            An assertion error is raised if the workflow was not successfully started.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/actions/workflows/{filename}/dispatches"
        json = {"ref": "devel",
                "inputs": inputs}
        print(url, json)
        reply = self._post_request("POST", url, json)
        print(reply.text)
        assert reply.status_code == 204

    def get_comments(self, pr_id):
        """
        Get all comments left on a given pull request or issue.

        Get a dictionary containing a list of all the comments left
        on a given pull request or issue. This list is obtained using
        the API as described here:
        https://docs.github.com/en/rest/issues/comments?apiVersion=2022-11-28

        Parameters
        ----------
        pr_id : int
            The id of the pull request or comment.

        Returns
        -------
        dict
            A dictionary containing the comments.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/issues/{pr_id}/comments"
        return self._post_request("GET", url).json()

    def get_review_comments(self, pr_id):
        """
        Get all review comments left on a given pull request.

        Get a dictionary containing a list of all the review comments left
        on a given pull request. This includes comments left as a reply to a
        review comment on a code snippet. This list is obtained using
        the API as described here:
        https://docs.github.com/en/rest/pulls/comments?apiVersion=2022-11-28

        Parameters
        ----------
        pr_id : int
            The id of the pull request or comment.

        Returns
        -------
        dict
            A dictionary containing the comments.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/pulls/{pr_id}/comments"
        return self._post_request("GET", url).json()

    def create_comment(self, pr_id, comment, reply_to = None):
        """
        Create a comment on a pull request or issue.

        Create a comment on a pull request or issue using the API as
        described here:
        https://docs.github.com/en/rest/pulls/comments?apiVersion=2022-11-28

        Comments can also be left as a reply to a review comment on
        a code snippet. This requires the id of the original comment
        to be provided.

        Parameters
        ----------
        pr_id : int
            The id of the pull request or comment.

        comment : str
            The message to be left in the comment.

        reply_to : int, optional
            The id of the comment being replied to.

        Returns
        -------
        requests.Response
            The response collected from the request.
        """
        if reply_to:
            suffix = f"/{reply_to}/replies"
            issue_type = 'pulls'
        else:
            issue_type = 'issues'
            suffix = ''
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/{issue_type}/{pr_id}/comments{suffix}"
        print(url)
        return self._post_request("POST", url, json={"body":comment})

    def create_review(self, pr_id, commit, comment, status, comments = ()):
        """
        Create a review on the specified pull request.

        Create a review on the specified pull request using the API as
        described here:
        https://docs.github.com/en/rest/pulls/reviews?apiVersion=2022-11-28#create-a-review-for-a-pull-request

        Parameters
        ----------
        pr_id : int
            The id of the pull request or comment.

        commit : str
            The SHA of the most recent commit at the moment of the review.

        comment : str
            The message to be left in the review.

        status : str
            The status of the review, [REQUEST_CHANGES/APPROVE].

        comments : list of dictionaries
            A list of dictionaries describing the comments to be left on code snippets.

        Returns
        -------
        requests.Response
            The response collected from the request.

        Raises
        ------
        AssertionError
            An assertion error is raised if the review was not successfully posted.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/pulls/{pr_id}/reviews"
        review = {'commit_id':commit, 'body': comment, 'event': status, 'comments': comments}
        print(review)
        reply = self._post_request("POST", url, json=review)
        print(reply.text)
        assert reply.status_code == 200
        return reply

    def check_for_user_in_team(self, user, team):
        """
        Check to determine if a user belongs to a given team.

        Use the API to check to determine if a user belongs to a given team
        as described here:
        https://docs.github.com/en/rest/teams/members?apiVersion=2022-11-28#get-team-membership-for-a-user

        Parameters
        ----------
        user : str
            The user of interest.

        team : str
            The team which we are checking.

        Returns
        -------
        dict
            A dictionary describing the result.
        """
        url = f'https://api.github.com/orgs/{self._org}/teams/{team}/membersips/{user}'
        return self._post_request("GET", url).json()

    def get_prs(self, state='open'):
        """
        Get a list of all pull requests in the repository.

        Use the API to get a list of all pull requests in the repository which have
        the specified state as described here:
        https://docs.github.com/en/rest/pulls/pulls?apiVersion=2022-11-28#list-pull-requests

        Parameters
        ----------
        state : str, default='open'
            The state of the pull requests to report [open/closed/all].

        Returns
        -------
        dict
            A dictionary describing the pull requests.
        """
        url = f'https://api.github.com/repos/{self._org}/{self._repo}/pulls'
        return self._post_request("GET", url).json()

    def get_check_runs(self, commit):
        """
        Get a list of all check runs which have run on a given commit.

        Use the API to get a list of all check runs which have run on a given
        commit as described here:
        https://docs.github.com/en/rest/checks/runs?apiVersion=2022-11-28#list-check-runs-for-a-git-reference

        Parameters
        ----------
        commit : str
            The SHA of the most recent commit at the moment of the review.

        Returns
        -------
        dict
            A dictionary describing the check runs.
        """
        url = f'https://api.github.com/repos/{self._org}/{self._repo}/commits/{commit}/check-runs'
        return self._post_request("GET", url).json()

    def get_pr_events(self, pr_id):
        """
        Get a list of all events which occured on this pull request.

        Use the API to get a list of all events which occured on this pull
        request as described here:
        https://docs.github.com/en/rest/issues/events?apiVersion=2022-11-28#list-issue-events

        Parameters
        ----------
        pr_id : int
            The id of the pull request.

        Returns
        -------
        dict
            A dictionary describing the events.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/issues/{pr_id}/events"
        return self._post_request("GET", url).json()

    def get_artifacts(self, name):
        """
        Find all artifacts with the specified name.

        Find all artifacts in the repository with the specified name using
        the API as described here:
        https://docs.github.com/en/rest/actions/artifacts?apiVersion=2022-11-28#list-artifacts-for-a-repository

        Parameters
        ----------
        name : str
            The name of the artifact.

        Returns
        -------
        dict
            A dictionary describing all artifacts found.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/actions/artifacts"
        query= {'name': name}
        return self._post_request("GET", url).json()

    def download_artifact(self, name, url):
        """
        Download the specified artifact from the url.

        Use the API to download the specified artifact from the url
        into a file called `name` as described here:
        https://docs.github.com/en/rest/actions/artifacts?apiVersion=2022-11-28#download-an-artifact

        Parameters
        ----------
        name : str
            The name of the file where the result should be saved.

        url : str
            The url where the file is located.
        """
        reply = self._post_request("GET", url, stream=True)
        with open(name, 'wb') as f:
            f.write(reply.content)

    def get_reviews(self, pr_id):
        """
        Get a list of all reviews which have been left on a given pull request.

        Use the API to get a list of all reviews left on a given pull request
        as described here:
        https://docs.github.com/en/rest/pulls/reviews?apiVersion=2022-11-28#list-reviews-for-a-pull-request

        Parameters
        ----------
        pr_id : int
            The id of the pull request.

        Returns
        -------
        dict
            A dictionary describing the reviews.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/pulls/{pr_id}/reviews"
        return self._post_request("GET", url).json()

    def get_events(self, pr_id):
        """
        Get a timeline of events which occured on a given pull request.

        Use the API to get a list of events on a pull request as described
        here:
        https://docs.github.com/en/rest/issues/timeline?apiVersion=2022-11-28

        These events are described here:
        https://docs.github.com/en/webhooks-and-events/events/issue-event-types

        Parameters
        ----------
        pr_id : int
            The id of the pull request.

        Returns
        -------
        dict
            A dictionary describing the events.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/issues/{pr_id}/timeline"
        return self._post_request("GET", url).json()

    def clear_labels(self, pr_id, labels):
        """
        Remove the specified labels from the indicated pull request.

        Use the API to remove the specified labels from the indicated pull
        request as described here:
        https://docs.github.com/en/rest/issues/labels?apiVersion=2022-11-28#remove-a-label-from-an-issue

        Parameters
        ----------
        pr_id : int
            The id of the pull request.

        labels : list of str
            A list containing the names of the labels to be removed.
        """
        for l in labels:
            url = f"https://api.github.com/repos/{self._org}/{self._repo}/issues/{pr_id}/labels/{l}"
            self._post_request("DELETE", url)

    def add_labels(self, pr_id, labels):
        """
        Add the specified labels to the indicated pull request.

        Use the API to add the specified labels from the indicated pull
        request as described here:
        https://docs.github.com/en/rest/issues/labels?apiVersion=2022-11-28#add-labels-to-an-issue

        Parameters
        ----------
        pr_id : int
            The id of the pull request.

        labels : list of str
            A list containing the names of the labels to be added.
        """
        assert labels
        url = f"https://api.github.com/repos/OWNER/REPO/issues/ISSUE_NUMBER/labels"
        self._post_request("POST", url, {"labels":labels})

    def get_current_labels(self, pr_id):
        """
        Get a description of all labels currently on the pull request.

        Use the API to get a description of all labels currently used
        on the specified pull request.

        Parameters
        ----------
        pr_id : int
            The id of the pull request.

        Returns
        -------
        list of dict
            A list of dictionaries describing each of the labels.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/issues/{pr_id}/labels"
        return self._post_request("POST", url).json()

    def get_headers(self):
        """
        Get the header which is always passed to the API.

        Get the header which must always be passed to the API to authentificate.
        This header specifies the identity of the bot and the reference version
        of the GitHub API.

        Returns
        -------
        dict
            The header which should be used in requests.
        """
        if self._install_token_exp < time.struct_time(time.gmtime()):
            self._install_token, expiry = get_authorization()
            self._install_token_exp = time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ")

        return {"Accept": "application/vnd.github+json",
                 "Authorization": f"Bearer {self._install_token}",
                 "X-GitHub-Api-Version": "2022-11-28"}
