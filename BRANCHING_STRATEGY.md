# Git Branching Strategy

## Branches

### Main Branches
- **`main`**: Production-ready code (protected, requires PR approval)
- **`develop`**: Integration branch for active development

### Supporting Branches
- **`feature/*`**: Feature development (e.g., `feature/package-structure`)
- **`bugfix/*`**: Bug fixes (e.g., `bugfix/fix-rsrp-validation`)
- **`release/*`**: Release preparation (e.g., `release/v0.1.0`)
- **`hotfix/*`**: Emergency production fixes (e.g., `hotfix/critical-bug`)

---

## Workflow

### Feature Development
```bash
# 1. Start from develop
git checkout develop
git pull origin develop

# 2. Create feature branch
git checkout -b feature/my-feature

# 3. Make changes and commit
git add .
git commit -m "feat: Add my feature"

# 4. Push and create Pull Request
git push -u origin feature/my-feature
# Then create PR in GitHub/GitLab: feature/my-feature → develop

# 5. After PR approval and merge, delete feature branch
git checkout develop
git pull origin develop
git branch -d feature/my-feature
```

### Release Process
```bash
# 1. Create release branch from develop
git checkout -b release/v0.1.0 develop

# 2. Update version numbers, changelog
# Make any final bug fixes

# 3. Merge to main
git checkout main
git merge --no-ff release/v0.1.0
git tag -a v0.1.0 -m "Release version 0.1.0"

# 4. Merge back to develop
git checkout develop
git merge --no-ff release/v0.1.0

# 5. Delete release branch
git branch -d release/v0.1.0
```

### Hotfix Process
```bash
# 1. Create hotfix from main
git checkout -b hotfix/critical-fix main

# 2. Fix the issue
git commit -m "fix: Critical bug fix"

# 3. Merge to main
git checkout main
git merge --no-ff hotfix/critical-fix
git tag -a v0.1.1 -m "Hotfix version 0.1.1"

# 4. Merge back to develop
git checkout develop
git merge --no-ff hotfix/critical-fix

# 5. Delete hotfix branch
git branch -d hotfix/critical-fix
```

---

## Commit Message Format

Follow **Conventional Commits** specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation changes
- **style**: Code style (formatting, no logic change)
- **refactor**: Code refactoring
- **perf**: Performance improvement
- **test**: Adding or updating tests
- **chore**: Maintenance tasks (dependencies, build, etc.)
- **ci**: CI/CD changes

### Examples

**Simple feature:**
```
feat: Add RSRP prediction for synthetic grids
```

**With scope:**
```
fix(geometry): Correct haversine distance calculation
```

**With body:**
```
feat: Implement IDW-based RSRP prediction

Uses k=8 nearest neighbors from same cell with inverse
distance weighting. Fallback to per-cell path loss model
if insufficient neighbors.
```

**With breaking change:**
```
refactor!: Change config schema to use Pydantic

BREAKING CHANGE: Old YAML configs need migration.
See migration guide in docs/migration.md
```

**With issue reference:**
```
fix: Validate RSRP range in data loader

Fixes #123
```

---

## Branch Protection Rules

### `main` Branch
- ✅ Require pull request reviews (minimum 1 approval)
- ✅ Require status checks to pass (tests, linting)
- ✅ No direct commits
- ✅ No force pushes
- ✅ Require linear history

### `develop` Branch
- ✅ Require pull request reviews (minimum 1 approval)
- ✅ Require status checks to pass
- ⚠️ Allow squash merging
- ❌ Allow force pushes (with caution)

---

## Pull Request Guidelines

### Before Creating PR
- [ ] All tests pass locally
- [ ] Code formatted (Black)
- [ ] Linting passes (flake8)
- [ ] Documentation updated
- [ ] Commit messages follow convention

### PR Template
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] Manual testing completed

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
- [ ] Tests pass locally
```

### Review Process
1. **Author** creates PR and requests review
2. **Reviewer(s)** provide feedback within 24 hours
3. **Author** addresses feedback
4. **Reviewer** approves
5. **Author** or **Maintainer** merges

---

## Tips

### Keep Branches Small
- Focus on one feature/fix per branch
- Aim for PRs < 500 lines changed
- Break large features into smaller PRs

### Sync Regularly
```bash
# Keep your feature branch up to date
git checkout feature/my-feature
git fetch origin
git rebase origin/develop
```

### Clean Up Old Branches
```bash
# List merged branches
git branch --merged develop

# Delete merged branches (except main/develop)
git branch -d old-feature-branch
```

### Interactive Rebase (Clean Commit History)
```bash
# Before creating PR, squash WIP commits
git rebase -i develop

# Mark commits to squash (s), reword (r), or drop (d)
```

---

## Questions?

Contact the tech lead or see [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.
