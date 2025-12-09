# Git Hooks

This project uses Git hooks to enforce code quality standards before commits and pushes.

## Initial Setup (Required)

After cloning the repository, **you must enable the hooks** by running:

```bash
git config core.hooksPath .githooks
```

This tells Git to use the version-controlled hooks in `.githooks/` instead of the default `.git/hooks/` directory.

**Note:** This is a one-time setup per repository clone. The hooks will then work automatically for all commits and pushes.

## Hooks Overview

### Pre-commit Hook
**Purpose**: Fast quality checks on every commit  
**Actions**:
- Runs `uv run ruff format .` (formats code first)
- Runs `uv run ruff check . --fix` (auto-fixes linting issues on formatted code)
- Auto-stages formatted files

**Performance**: Fast (~few seconds), skips test execution

### Pre-push Hook
**Purpose**: Comprehensive validation before pushing to remote  
**Actions**:
- Runs `uv run ruff format .` (formatting first)
- Runs `uv run ruff check . --fix` (linting on formatted code)
- Runs `uv run pytest tests/ -v --tb=line -m "not gpu"` (all non-GPU tests)

**Performance**: Slower (~depends on test suite), ensures code quality

## Usage

### Normal Workflow
The hooks run automatically:

```bash
# Commit triggers pre-commit hook
git commit -m "your message"

# Push triggers pre-push hook
git push
```

If any check fails, the commit/push will be blocked with a clear error message.

### Bypass Hooks (Emergency Only)
Use `--no-verify` flag to skip hooks when absolutely necessary:

```bash
# Skip pre-commit checks
git commit --no-verify -m "emergency fix"

# Skip pre-push checks
git push --no-verify
```

**Warning**: Only use `--no-verify` in emergencies. The CI pipeline will still enforce these checks.

## What Gets Checked

### Linting (Ruff)
- Code style compliance (E, F, W, I, B, UP, C4 rules)
- Import sorting
- Type annotation requirements (for public functions)
- Configured in `pyproject.toml` under `[tool.ruff]`

### Formatting (Ruff)
- Consistent code formatting
- Line length: 100 characters
- Double quotes for strings
- 4-space indentation

### Testing (pytest)
- All tests in `tests/` directory
- Excludes GPU-marked tests (`-m "not gpu"`)
- Verbose output with line-level tracebacks
- Only runs on pre-push (not pre-commit)

## Troubleshooting

### Hook Not Running
If hooks aren't running, verify they exist and are executable:

```bash
# List hooks
ls -la .git/hooks/

# On Windows with Git Bash
ls -la .git/hooks/pre-commit .git/hooks/pre-push
```

### Ruff Errors
If Ruff reports errors that can't be auto-fixed:
1. Read the error message carefully
2. Fix the issue manually
3. Stage the changes
4. Try committing again

### Test Failures
If tests fail during pre-push:
1. Run tests locally: `uv run pytest tests/ -v --tb=line -m "not gpu"`
2. Fix the failing tests
3. Ensure all tests pass locally
4. Try pushing again

### Performance Issues
If pre-push is too slow:
- Consider using `--no-verify` for WIP branches (but fix before final merge)
- Or temporarily disable GPU tests marker in your local environment
- Remember: CI will run full test suite anyway

## Maintenance

### Updating Hook Scripts
Hooks are located in `.git/hooks/`:
- `.git/hooks/pre-commit` - Pre-commit checks
- `.git/hooks/pre-push` - Pre-push checks

After modifying hooks, they take effect immediately (no installation needed).

### Disabling Hooks Permanently (Not Recommended)
To disable hooks project-wide:

```bash
# Remove or rename the hooks
mv .git/hooks/pre-commit .git/hooks/pre-commit.disabled
mv .git/hooks/pre-push .git/hooks/pre-push.disabled
```

**Note**: This is strongly discouraged. Hooks exist to catch issues early.
