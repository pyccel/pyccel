from .. import hookimpl
from .semantic import _visit_CodeBlock

@hookimpl
def get_description():
    return "Add comments in generated code indicating which line in the Python file corresponds to the generated code."

@hookimpl
def add_cli_options(parser, cli_tool):
    pass

@hookimpl
def get_updated_semantic_methods():
    return (_visit_CodeBlock,)
