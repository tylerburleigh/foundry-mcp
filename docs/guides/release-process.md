# Release Process

This document describes the release process for foundry-mcp, including stable and beta releases.

## Branch Strategy

```
main ─────────────────────────────────────────► (stable releases)
   │
   └── beta ──────────────────────────────────► (beta releases)
           │
           └── feature branches
```

| Branch | Purpose | Publishes To |
|--------|---------|--------------|
| `main` | Stable releases | PyPI (stable), GitHub Releases |
| `beta` | Beta/pre-release features | PyPI (pre-release), GitHub Pre-releases |

## Version Numbering

We follow [PEP 440](https://peps.python.org/pep-0440/) for version numbering, which is required for PyPI compatibility.

### Stable Releases

Format: `MAJOR.MINOR.PATCH` (e.g., `0.8.36`, `1.0.0`)

### Beta Releases

Format: `MAJOR.MINOR.PATCHbN` where N is the beta number.

| Version | Meaning |
|---------|---------|
| `0.9.0b1` | First beta of 0.9.0 |
| `0.9.0b2` | Second beta of 0.9.0 |
| `0.9.0rc1` | Release candidate 1 of 0.9.0 |
| `0.9.0` | Final stable release |

### Other Pre-release Types

| Suffix | Meaning | Example |
|--------|---------|---------|
| `aN` | Alpha release | `0.9.0a1` |
| `bN` | Beta release | `0.9.0b1` |
| `rcN` | Release candidate | `0.9.0rc1` |
| `.devN` | Development release | `0.9.0.dev1` |

## Release Workflows

### Creating a Stable Release

1. **Ensure all changes are merged to `main`**

2. **Update version in `pyproject.toml`**:
   ```toml
   version = "0.8.37"
   ```

3. **Update CHANGELOG.md**:
   - Move items from `[Unreleased]` to new version section
   - Add release date

4. **Commit and tag**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: v0.8.37"
   git tag v0.8.37
   git push origin main --tags
   ```

5. **Create GitHub Release**:
   - Go to Releases → Draft a new release
   - Select the tag
   - Generate release notes
   - Publish release

6. The `publish.yml` workflow will automatically publish to PyPI.

### Creating a Beta Release

1. **Switch to beta branch**:
   ```bash
   git checkout beta
   git pull origin beta
   ```

2. **Merge or cherry-pick changes**:
   ```bash
   # Option A: Merge from a feature branch
   git merge feature/my-feature

   # Option B: Cherry-pick specific commits
   git cherry-pick abc123
   ```

3. **Update version to beta format**:
   ```toml
   version = "0.9.0b1"  # First beta of 0.9.0
   ```

4. **Update CHANGELOG.md** (use `[0.9.0b1]` heading)

5. **Commit and push**:
   ```bash
   git add pyproject.toml CHANGELOG.md
   git commit -m "release: v0.9.0-beta.1"
   git push origin beta
   ```

6. **Tag and push** (optional, for GitHub release):
   ```bash
   git tag v0.9.0-beta.1
   git push origin v0.9.0-beta.1
   ```

7. The `publish-beta.yml` workflow will:
   - Validate it's a beta version
   - Build and publish to PyPI
   - Create a GitHub pre-release (if tagged)

### Promoting Beta to Stable

When a beta is ready for stable release:

1. **Switch to main**:
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Merge beta branch**:
   ```bash
   git merge beta
   ```

3. **Update version to stable**:
   ```toml
   version = "0.9.0"  # Remove beta suffix
   ```

4. **Update CHANGELOG.md**:
   - Consolidate beta entries under `[0.9.0]`
   - Add release date

5. **Follow stable release process** (steps 4-6 above)

## Installing Pre-releases

Users can install beta versions:

```bash
# Install specific beta version
pip install foundry-mcp==0.9.0b1

# Install latest pre-release (including betas)
pip install --pre foundry-mcp

# Install latest stable only (default)
pip install foundry-mcp
```

## GitHub Environments

The following GitHub environments are used:

| Environment | Purpose | Required Secrets |
|-------------|---------|------------------|
| `pypi` | PyPI publishing | Trusted publisher (no secrets needed) |

## Troubleshooting

### "Version is not a beta version" error

The version in `pyproject.toml` must contain a PEP 440 pre-release suffix:
- ✅ `0.9.0b1`, `0.9.0a1`, `0.9.0rc1`, `0.9.0.dev1`
- ❌ `0.9.0-beta.1`, `0.9.0-beta1`, `0.9.0`

### PyPI rejects the version

Ensure the version follows [PEP 440](https://peps.python.org/pep-0440/). Common issues:
- Hyphens are not allowed (use `b` not `-beta.`)
- Version must be unique (can't re-upload same version)

### Workflow doesn't trigger

Check that:
- You're pushing to the correct branch (`beta` for betas)
- The tag format matches the workflow trigger patterns
- You have the required permissions
