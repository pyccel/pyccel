"""File containing all functions useful for handling interactions with the GitHub API."""

import os
import time

import jwt
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
    payload = {"iat": int(time.time()), "exp": int(time.time()) + 60, "iss": 364561}

    jw_token = jwt.JWT().encode(payload, signing_key, alg="RS256")

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {jw_token}",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Create JWT
    reply = requests.post(
        "https://api.github.com/app/installations/39885334/access_tokens",
        headers=headers,
        timeout=10,
    )

    print(reply.text)

    json_reply = reply.json()

    print(json_reply)

    token = json_reply["token"]
    expiry = json_reply["expires_at"]

    with open(os.environ["GITHUB_ENV"], "r", encoding="utf-8") as f:
        output = f.read()

    if "installation_token" in output:
        lines = output.split("\n")
        print("Parsed : ", lines)
        output = "\n".join(l for l in lines if "installation_token" not in l)

    with open(os.environ["GITHUB_ENV"], "w", encoding="utf-8") as f:
        f.write(output)
        print(f"installation_token={token}", file=f)
        print(f"installation_token_exp={expiry}", file=f)

    return token, expiry


class GitHubAPIInteractions:
    """
    Class which handles all interactions with the GitHub API.

    A helper class which exposes the GitHub API in a readable
    manner.

    Parameters
    ----------
    repo : str
        A string which identifies the repository where the requests
        should be made (e.g. 'pyccel/pyccel').
    """

    def __init__(self, repo):
        repo = repo or os.environ["GITHUB_REPOSITORY"]
        self._org, self._repo = repo.split("/")
        if "installation_token" in os.environ:
            self._authenticated = True
            self._install_token = os.environ["installation_token"]
            self._install_token_exp = time.strptime(
                os.environ["installation_token_exp"], "%Y-%m-%dT%H:%M:%SZ"
            )
        elif "PEM" in os.environ:
            self._authenticated = True
            self._install_token, expiry = get_authorization()
            self._install_token_exp = time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ")
        else:
            self._authenticated = False

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
        reply = requests.request(
            method, url, json=json, headers=self.get_headers(), timeout=10, **kwargs
        )
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

    def post_coverage_run(self, commit, name, json):
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

        json : dictionary
            The information that should be updated in the check run.

        Returns
        -------
        requests.Response
            The response collected from the request.

        Raises
        ------
        AssertionError
            An assertion error is raised if the check run was not successfully posted.
        """
        assert self._authenticated
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/check-runs"
        workflow_url = f"https://github.com/{self._org}/{self._repo}/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        print("create_run:", url)
        json.update(
            {
                "name": name,
                "head_sha": commit,
                "status": "in_progress",
                "details_url": workflow_url,
            }
        )
        run = self._post_request("POST", url, json)
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
        results = []
        page = 1
        new_results = [None]
        while len(new_results) != 0:
            request = self._post_request(
                "GET", url, params={"per_page": "100", "page": str(page)}
            )
            new_results = request.json()
            results.extend(new_results)
            page += 1
        return results

    def create_comment(self, pr_id, comment, reply_to=None):
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
        assert self._authenticated
        if reply_to:
            suffix = f"/{reply_to}/replies"
            issue_type = "pulls"
        else:
            issue_type = "issues"
            suffix = ""
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/{issue_type}/{pr_id}/comments{suffix}"
        print(url)
        return self._post_request("POST", url, json={"body": comment})

    def create_review(self, pr_id, commit, comment, status, comments=()):
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
        assert self._authenticated
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/pulls/{pr_id}/reviews"
        review = {
            "commit_id": commit,
            "body": comment,
            "event": status,
            "comments": comments,
        }
        print(review)
        reply = self._post_request("POST", url, json=review)
        print(reply.text)
        assert reply.status_code == 200
        return reply

    def wait_for_runs(self, commit_sha, run_names, max_time=15 * 60, wait_time=15):
        """
        Wait for the specified workflow runs associated with the commit.

        Use the API to find workflow runs associated with a commit as described here:
        https://docs.github.com/en/rest/actions/workflow-runs?apiVersion=2022-11-28#list-workflow-runs-for-a-repository

        If the workflows are not finished then wait before trying again. Keep trying until the
        timeout is reached.

        Parameters
        ----------
        commit_sha : str
            The SHA of the commit for which the workflows are run.

        run_names : tuple[str,...]
            The names of the workflows which should finish before this method terminates.

        max_time : int, default=15*60
            The maximum number of seconds which should be spent in this method.

        wait_time : int, default=15
            The time in seconds that the method should wait for before rechecking the workflow
            results.

        Returns
        -------
        bool
            True if all workflows succeeded, False otherwise.
        """
        url = f"https://api.github.com/repos/{self._org}/{self._repo}/actions/runs"
        completed = False
        # Run for 15 mins maximum
        timeout = time.time() + max_time

        request_result = self._post_request(
            "GET", url, params={"head_sha": commit_sha}
        ).json()
        workflow_runs = [
            j for j in request_result["workflow_runs"] if j["name"] in run_names
        ]
        completed = all(j["status"] == "completed" for j in workflow_runs)

        print(commit_sha)
        print(workflow_runs)
        print([j["name"] for j in workflow_runs])
        print([j["status"] for j in workflow_runs])
        print([j["conclusion"] for j in workflow_runs])

        while not completed and time.time() < timeout:
            time.sleep(15)
            request_result = self._post_request(
                "GET", url, params={"head_sha": commit_sha}
            ).json()
            workflow_runs = [
                j for j in request_result["workflow_runs"] if j["name"] in run_names
            ]
            completed = all(j["status"] == "completed" for j in workflow_runs)
            print(workflow_runs)
            print([j["name"] for j in workflow_runs])
            print([j["status"] for j in workflow_runs])
            print([j["conclusion"] for j in workflow_runs])

        assert completed

        success = all(j["conclusion"] == "success" for j in workflow_runs)

        return success

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
        if self._authenticated:
            if self._install_token_exp < time.struct_time(time.gmtime()):
                self._install_token, expiry = get_authorization()
                self._install_token_exp = time.strptime(expiry, "%Y-%m-%dT%H:%M:%SZ")

            return {
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._install_token}",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        else:
            return {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
