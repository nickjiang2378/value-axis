"""Shared code analysis utilities for code steering benchmarks (10 & 11)."""

import ast
import random
import re
import string


def extract_code_block(text):
    """Extract the first ```python``` code block from text. Falls back to full text."""
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Try generic code block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def count_lines(code):
    """Count non-empty lines in code."""
    return len([line for line in code.splitlines() if line.strip()])


def count_comments(code):
    """Count comment lines (lines starting with # after stripping)."""
    count = 0
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            count += 1
    return count


def count_type_hints(code):
    """Count type hint annotations in code using AST parsing."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        # Fallback to regex
        return len(re.findall(r":\s*(?:int|str|float|bool|list|dict|tuple|set|Optional|Union|List|Dict|Tuple|Set|Any|None)\b", code))

    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            # Return annotation
            if node.returns is not None:
                count += 1
            # Argument annotations (skip 'self')
            for arg in node.args.args:
                if arg.annotation is not None and arg.arg != "self":
                    count += 1
        elif isinstance(node, ast.AnnAssign):
            count += 1
    return count


def check_syntax(code):
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def code_length(code):
    """Return the character length of code."""
    return len(code)


def analyze_code(text):
    """Extract code from response and compute all metrics."""
    code = extract_code_block(text)
    return {
        "code": code,
        "n_lines": count_lines(code),
        "n_comments": count_comments(code),
        "n_type_hints": count_type_hints(code),
        "syntax_valid": check_syntax(code),
        "code_len": code_length(code),
    }


# ── Code transformations for Benchmark 11 ──────────────────────────────────


def introduce_syntax_error(code, seed=42):
    """Introduce one random syntax error: missing colon, unmatched paren, or bad indent.

    Finds all applicable error types for the given code, then randomly picks one.
    Returns the corrupted code string. Deterministic given seed.
    """
    rng = random.Random(seed)
    lines = code.splitlines()
    if not lines:
        return code

    # Find all applicable error types
    applicable = []

    colon_lines = [i for i, line in enumerate(lines) if line.rstrip().endswith(":")]
    if colon_lines:
        applicable.append("missing_colon")

    paren_lines = [i for i, line in enumerate(lines) if "(" in line]
    if paren_lines:
        applicable.append("unmatched_paren")

    indented = [i for i, line in enumerate(lines) if line and line[0] == " "]
    if indented:
        applicable.append("bad_indent")

    if not applicable:
        return code

    error_type = rng.choice(applicable)

    if error_type == "missing_colon":
        idx = rng.choice(colon_lines)
        lines[idx] = lines[idx].rstrip().rstrip(":") + lines[idx][len(lines[idx].rstrip()):]
        return "\n".join(lines)

    if error_type == "unmatched_paren":
        idx = rng.choice(paren_lines)
        pos = lines[idx].index("(")
        lines[idx] = lines[idx][:pos] + "((" + lines[idx][pos + 1:]
        return "\n".join(lines)

    if error_type == "bad_indent":
        idx = rng.choice(indented)
        lines[idx] = lines[idx].lstrip()
        return "\n".join(lines)

    return code


def shuffle_lines(code, seed=42):
    """Shuffle body lines while preserving header (imports, def signature).

    Preserves: lines before the first function/class definition body.
    Shuffles: everything after the first indented block starts.
    """
    rng = random.Random(seed)
    lines = code.splitlines()
    if len(lines) <= 2:
        return code

    # Find where the body starts (first indented line after a def/class/if)
    body_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and (stripped.startswith("def ") or stripped.startswith("class ")
                         or stripped.startswith("if ") or stripped.startswith("for ")
                         or stripped.startswith("while ")):
            # Next line should be indented body
            if i + 1 < len(lines):
                body_start = i + 1
                break

    if body_start is None:
        # No clear structure, shuffle everything after line 1
        body_start = 1

    header = lines[:body_start]
    body = lines[body_start:]
    rng.shuffle(body)
    return "\n".join(header + body)


def obfuscate_variables(code, seed=42):
    """Rename local variables to single letters.

    Uses AST to find variable names, replaces them with sequential letters.
    Falls back to regex if AST parsing fails.
    """
    rng = random.Random(seed)

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return obfuscate_regex(code, rng)

    # Collect variable names (assignments, function args)
    var_names = set()
    builtin_names = set(dir(__builtins__)) if isinstance(__builtins__, dict) else set(dir(__builtins__))
    # Add common stdlib names we don't want to rename
    protected = builtin_names | {
        "self", "cls", "args", "kwargs", "stdin", "stdout", "stderr",
        "sys", "os", "re", "math", "json", "collections", "itertools",
        "functools", "heapq", "bisect", "defaultdict", "deque", "Counter",
        "List", "Dict", "Tuple", "Set", "Optional", "Union", "Any",
        "True", "False", "None", "print", "input", "range", "len",
        "int", "str", "float", "bool", "list", "dict", "tuple", "set",
        "sorted", "reversed", "enumerate", "zip", "map", "filter",
        "min", "max", "sum", "abs", "all", "any", "isinstance",
        "hasattr", "getattr", "setattr", "type", "super", "open",
        "Solution", "TreeNode", "ListNode",  # LeetCode common classes
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name not in protected:
                var_names.add(node.name)
            for arg in node.args.args:
                if arg.arg not in protected:
                    var_names.add(arg.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if node.id not in protected:
                var_names.add(node.id)

    if not var_names:
        return code

    # Create mapping to single letters
    letters = list(string.ascii_lowercase)
    rng.shuffle(letters)
    mapping = {}
    for i, name in enumerate(sorted(var_names)):
        if i < len(letters):
            mapping[name] = letters[i]
        else:
            mapping[name] = letters[i % len(letters)] + str(i // len(letters))

    # Apply substitutions using word boundaries
    result = code
    for old_name, new_name in sorted(mapping.items(), key=lambda x: -len(x[0])):
        result = re.sub(r"\b" + re.escape(old_name) + r"\b", new_name, result)

    return result


def obfuscate_regex(code, rng):
    """Regex fallback for obfuscation when AST fails."""
    # Find variable-like assignments
    var_pattern = re.compile(r"\b([a-z_][a-z0-9_]*)\s*=", re.IGNORECASE)
    matches = var_pattern.findall(code)

    protected = {
        "self", "cls", "True", "False", "None", "print", "input", "range",
        "len", "int", "str", "float", "bool", "list", "dict", "return",
        "if", "else", "elif", "for", "while", "def", "class", "import",
        "from", "in", "not", "and", "or", "is", "with", "as", "try",
        "except", "finally", "raise", "pass", "break", "continue",
        "Solution", "TreeNode", "ListNode",
    }

    var_names = [m for m in set(matches) if m not in protected]
    if not var_names:
        return code

    letters = list(string.ascii_lowercase)
    rng.shuffle(letters)
    mapping = {}
    for i, name in enumerate(sorted(var_names)):
        if i < len(letters):
            mapping[name] = letters[i]
        else:
            mapping[name] = letters[i % len(letters)] + str(i // len(letters))

    result = code
    for old_name, new_name in sorted(mapping.items(), key=lambda x: -len(x[0])):
        result = re.sub(r"\b" + re.escape(old_name) + r"\b", new_name, result)

    return result
