# Getting Started

In order to get started contributing to Pyccel it is important to first ensure that you have installed Pyccel from source with an editable installation as described in the [installation docs](../docs/installation.md#from-sources).
If successful you should be able to add a print inside a function that is always used (e.g. in `pyccel/codegen/pipeline.py`) and see that it is correctly printed when the `pyccel` command line tool is used to translate an example file.

Once you have confirmed that you have a working editable installation you should be able to make changes to the code in a branch or fork to fix your chosen issue.

If this is your first contribution, then we recommend starting with an issue labelled as a `good-first-issue`. These issues roughly fall into two groups:
1.  New features whose implementation should be carried out similarly to an existing feature.
2.  Issues where the work is confined to only one of Pyccel's stages (see the [overview](./overview.md) for a definition of the different stages).

If it is not clear which case your chosen issue falls into, don't hesitate to get in touch on the issue or via the [Pyccel Discord Server](https://discord.gg/2Q6hwjfFVb).

If your issue falls into the second group then check out the documentation for the relevant stage.

Before tackling your issue, don't forget to check out the [conventions](./development_conventions.md) used in the Pyccel project.
