# Review Process

The review process is the process through which a branch which solves an issue is merged into the master branch.

When you believe your branch is ready to merge you should create a pull request. Be sure to add a description which allows other developers to understand what your changes aim to do. You may also want to include a commit summary as the pull request description forms the basis of the commit message shown on the master branch. In addition you should make sure that your pull request links to the issue that it is solving so that issue is automatically closed when the pull request is merged.

Once the pull request is opened 4 tests should be triggered they are: 

-   **Linux** : Runs the suite of tests on a linux machine
-   **MacOS** : Runs the suite of tests on a mac os machine
-   **Windows** : Runs the suite of tests on a windows machine
-   **Codacy** : Runs a static compiler via the [codacy](https://app.codacy.com/gh/pyccel/pyccel/dashboard) platform

Once the pull request is open the tests will be triggered every time you push new changes to the branch. Please be mindful of this and try to avoid pushing multiple times in a row to save compute resources. If you do find you need to push repeatedly, don't hesitate to cancel concurrent jobs using the github "Actions" tab.

When the pull request is ready for review (ie. you are happy with it, and it is passing all tests) it can be marked as such and the review process can begin. This process is split into 3 stages which each have an associated label.

## Needs Initial Review

To request the first stage of the review process you should add the label `needs_inital_review` to your pull request. This asks for a review from anyone. The aim is to review the Python code and ensure that it is clean. New developers are encouraged to review any pull requests marked `needs_inital_review` as the process of understanding how developers integrate their improvements into the existing codebase can be quite instructive when getting to grips with the code. Examples of things to look out for in your review are:

-   Unclear comments/docstrings
-   Missing/Incomplete tests
-   Code which could be faster (e.g. use of loops instead of list comprehensions)
-   Lack of `__slots__`
-   Unnecessary code duplication

Once the initial reviewer is happy with the branch they should accept the pull request and change the label from `needs_inital_review` to `Ready_for_review`

## Ready for Review

By this stage the code should be quite clean and should be mostly ready to merge. This label indicates that it is now ready for a senior developer to review. All of our senior developers are volunteers with busy schedules so the review has been split in two in this way to allow pull requests to be reviewed faster. By only requiring that senior developers review once the code is already somewhat cleaned up we avoid the small number of senior developers compared to junior developers becoming a bottleneck for merging.

Senior developers will look for anything missed in the initial review. They will also ensure that the pull request makes sense in the context of planned future developments for Pyccel.

Once the senior developer is happy with the branch they should accept the pull request and change the label from `Ready_for_review` to `Ready_to_merge`

## Ready to merge

Once the code has been accepted by both a junior and a senior developer it should be ready to merge. This flag therefore indicates that one of our developers with merge permissions can review the code. They will look for anything missed by the previous two reviews.

Anyone can make silly mistakes so Pyccel aims to have all pull requests be reviewed by at least 2 developers before being merged to master.
