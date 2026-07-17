"""
One-off migration script for tests/epyccel/recognised_functions/test_numpy_funcs.py.

Moves locally-defined functions that are compiled once per test into
tests/epyccel/modules/numpy_funcs.py, so that the whole module can be compiled
once per language (via epyccel_module_with_fallback) instead of once per test.

A test is only migrated if:
  - its only argument is `language`,
  - it is not parametrized (e.g. @pytest.mark.parametrize),
  - `language` is not used anywhere except as the `language=language` keyword
    of an `epyccel(...)` call,
  - every `epyccel(...)` call in its body is a simple
    `var = epyccel(local_func, language=language)` (optionally `verbose=...`)
    assignment, where `local_func` is a function defined directly in the
    test's body.

Run with `--dry-run` to only print what would change.
"""

import argparse
import ast
import re
import sys

TEST_FILE = "recognised_functions/test_numpy_funcs.py"
MODULE_FILE = "recognised_functions/modules/numpy_funcs.py"
MODULE_NAME = "numpy_funcs"
FIXTURE_NAME = f"epyc_{MODULE_NAME}_mod"


class Skip(Exception):
    """Raised to reject a test function from migration, with a reason."""


def offsets(src):
    starts = [0]
    for line in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(line))

    def to_offset(lineno, col):
        return starts[lineno - 1] + col

    return to_offset


def node_span(off, node):
    return off(node.lineno, node.col_offset), off(node.end_lineno, node.end_col_offset)


def is_epyccel_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "epyccel"
    )


def is_typevar_assign(node):
    return (
        isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "TypeVar"
    )


def has_parametrize(func):
    for dec in func.decorator_list:
        target = dec.func if isinstance(dec, ast.Call) else dec
        name = ast.unparse(target)
        if name.endswith("parametrize"):
            return True
    return False


def annotation_names(fdef):
    names = set()
    for a in (
        list(fdef.args.args) + list(fdef.args.posonlyargs) + list(fdef.args.kwonlyargs)
    ):
        if isinstance(a.annotation, ast.Name):
            names.add(a.annotation.id)
    if isinstance(fdef.returns, ast.Name):
        names.add(fdef.returns.id)
    return names


def classify(func):
    """Return a dict describing how to migrate `func`, or raise Skip."""
    args = func.args
    if (
        args.posonlyargs
        or args.kwonlyargs
        or args.vararg
        or args.kwarg
        or args.defaults
        or args.kw_defaults
    ):
        raise Skip("unexpected argument shape")
    if [a.arg for a in args.args] != ["language"]:
        raise Skip("arguments are not exactly (language,)")
    if has_parametrize(func):
        raise Skip("parametrized test")

    # `language` must only be used as the `language=language` kwarg of an epyccel() call.
    all_epyccel_calls = [n for n in ast.walk(func) if is_epyccel_call(n)]
    language_kwarg_nodes = set()
    for call in all_epyccel_calls:
        for kw in call.keywords:
            if (
                kw.arg == "language"
                and isinstance(kw.value, ast.Name)
                and kw.value.id == "language"
            ):
                language_kwarg_nodes.add(id(kw.value))
    for n in ast.walk(func):
        if (
            isinstance(n, ast.Name)
            and n.id == "language"
            and isinstance(n.ctx, ast.Load)
        ):
            if id(n) not in language_kwarg_nodes:
                raise Skip("`language` used outside of epyccel(..., language=language)")

    # Every epyccel() call must be `var = epyccel(local_func, language=language[, verbose=...])`.
    assigns_with_epyccel = [
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Assign) and is_epyccel_call(n.value)
    ]
    if len(assigns_with_epyccel) != len(all_epyccel_calls):
        raise Skip("epyccel() call not in the form `var = epyccel(...)`")

    local_defs = {
        stmt.name: stmt for stmt in func.body if isinstance(stmt, ast.FunctionDef)
    }

    targets = {}  # def_node id -> {"def_node":..., "calls": [assign_node, ...]}
    for assign in assigns_with_epyccel:
        call = assign.value
        if len(assign.targets) != 1 or not isinstance(assign.targets[0], ast.Name):
            raise Skip("epyccel() result not assigned to a single name")
        if len(call.args) != 1 or not isinstance(call.args[0], ast.Name):
            raise Skip("epyccel() first argument is not a simple name")
        allowed_kwargs = {"language", "verbose"}
        kwarg_names = {kw.arg for kw in call.keywords}
        if not kwarg_names <= allowed_kwargs or "language" not in kwarg_names:
            raise Skip(f"unsupported epyccel() kwargs {kwarg_names}")

        fname = call.args[0].id
        if fname not in local_defs:
            raise Skip(f"epyccel() target '{fname}' is not a local nested function")
        def_node = local_defs[fname]
        targets.setdefault(id(def_node), {"def_node": def_node, "calls": []})
        targets[id(def_node)]["calls"].append(assign)

    if not targets:
        raise Skip("no eligible epyccel() calls")

    # Attach a directly-preceding TypeVar preamble if it's referenced by the def's annotations.
    body = func.body
    for i, stmt in enumerate(body):
        if not isinstance(stmt, ast.FunctionDef) or id(stmt) not in targets:
            continue
        preamble = None
        if i > 0 and is_typevar_assign(body[i - 1]):
            tv_name = body[i - 1].targets[0].id
            if tv_name in annotation_names(stmt):
                preamble = body[i - 1]
        targets[id(stmt)]["preamble"] = preamble

    return {"targets": list(targets.values())}


def stripped_test_name(test_name):
    assert test_name.startswith("test_")
    return test_name[len("test_") :]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    src = open(TEST_FILE, encoding="utf-8").read()
    tree = ast.parse(src)
    off = offsets(src)

    test_funcs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    ]

    # Global (module-level, outside any test) TypeVar definitions: always copied verbatim.
    global_typevars = [n for n in tree.body if is_typevar_assign(n)]
    global_typevar_names = {n.targets[0].id for n in global_typevars}

    plans = []
    skips = []
    for func in test_funcs:
        try:
            plan = classify(func)
        except Skip as exc:
            skips.append((func.name, str(exc)))
            continue
        plan["func"] = func
        plans.append(plan)

    # Assign new names, checking for collisions.
    used_names = set()
    for func in tree.body:
        if isinstance(func, ast.FunctionDef):
            used_names.add(func.name)

    for plan in plans:
        func = plan["func"]
        stripped = stripped_test_name(func.name)
        multi = len(plan["targets"]) > 1
        for t in plan["targets"]:
            orig_name = t["def_node"].name
            new_name = f"{stripped}__{orig_name}" if multi else stripped
            if new_name in used_names:
                print(
                    f"ERROR: name collision hoisting '{orig_name}' from "
                    f"'{func.name}' -> '{new_name}' already exists",
                    file=sys.stderr,
                )
                sys.exit(1)
            used_names.add(new_name)
            t["new_name"] = new_name

    print(f"{len(plans)} tests migrated, {len(skips)} left untouched", file=sys.stderr)
    if args.dry_run:
        for plan in plans:
            func = plan["func"]
            for t in plan["targets"]:
                print(
                    f"  {func.name}: {t['def_node'].name} -> {MODULE_NAME}.{t['new_name']}"
                )
        return

    # --- Build the hoisted module content -----------------------------------
    module_lines = [
        "# pylint: disable=missing-function-docstring, missing-module-docstring\n"
    ]
    module_lines.append("from typing import TypeVar\n\n")
    module_lines.append("import numpy as np\n\n")
    for n in global_typevars:
        s, e = node_span(off, n)
        module_lines.append(src[s:e] + "\n")
    module_lines.append("\n")

    for plan in plans:
        for t in plan["targets"]:
            if t["preamble"] is not None:
                tv = t["preamble"]
                s, e = node_span(off, tv)
                module_lines.append(src[s:e] + "\n")
            def_node = t["def_node"]
            s, e = node_span(off, def_node)
            text = src[s:e]
            text = re.sub(
                rf"^def {re.escape(def_node.name)}\(", f"def {t['new_name']}(", text
            )
            module_lines.append(text + "\n\n")

    # --- Build the edits to the test file ------------------------------------
    edits = []  # (start, end, replacement)

    for plan in plans:
        func = plan["func"]
        # 1) function signature: language -> fixture
        lang_arg = func.args.args[0]
        s, e = node_span(off, lang_arg)
        edits.append((s, e, FIXTURE_NAME))

        # 2) remove hoisted defs (+ preamble), replace epyccel() calls, insert rebinds
        rebind_lines = []
        for t in plan["targets"]:
            block_start_node = (
                t["preamble"] if t["preamble"] is not None else t["def_node"]
            )
            s, e = node_span(off, block_start_node)
            s2, e2 = node_span(off, t["def_node"])
            edits.append((s, e2, None))  # delete the block entirely

            for call_assign in t["calls"]:
                cs, ce = node_span(off, call_assign.value)
                edits.append((cs, ce, f"{FIXTURE_NAME}.{t['new_name']}"))

            rebind_lines.append(f"{t['def_node'].name} = {MODULE_NAME}.{t['new_name']}")

        body = func.body
        insert_idx = 0
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            insert_idx = 1
        insert_node = body[insert_idx]
        insert_off = off(insert_node.lineno, 0)
        indent = " " * insert_node.col_offset
        text = "".join(f"{indent}{line}\n" for line in rebind_lines)
        edits.append((insert_off, insert_off, text))

    # 3) insert imports + fixture after the last top-level import
    last_import = None
    for n in tree.body:
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            last_import = n
    assert last_import is not None
    _, import_end = node_span(off, last_import)
    fixture_block = (
        "\nfrom epyccel_utilities import epyccel_module_with_fallback\n"
        "from modules import numpy_funcs\n\n\n"
        f'@pytest.fixture(scope="module")\n'
        f"def {FIXTURE_NAME}(language):\n"
        f"    return epyccel_module_with_fallback(numpy_funcs, language)\n\n"
    )
    edits.append((import_end, import_end, fixture_block))

    # apply edits back-to-front so earlier offsets stay valid
    edits.sort(key=lambda e: (-e[0], -e[1]))
    result = src
    for start, end, replacement in edits:
        result = result[:start] + (replacement or "") + result[end:]

    ast.parse(result)  # sanity check before writing

    with open(TEST_FILE, "w", encoding="utf-8") as f:
        f.write(result)
    with open(MODULE_FILE, "w", encoding="utf-8") as f:
        f.write("".join(module_lines))

    print(f"Wrote {TEST_FILE} and {MODULE_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
