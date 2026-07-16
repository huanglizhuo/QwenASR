# QwenASR Automated Release Process

This repository uses [Release Please](https://github.com/googleapis/release-please) to version and release three independently tracked components:

| Component | Registry or artifact | Tag format |
| --- | --- | --- |
| `crates/qwen-asr` | crates.io package `qwen-asr` | `qwen-asr-vX.Y.Z` |
| `crates/qwen-asr-cli` | crates.io package `qwen-asr-cli` and GitHub release binaries | `qwen-asr-cli-vX.Y.Z` |
| `flutter/qwen_asr` | pub.dev package `qwen_asr` | `qwen_asr-vX.Y.Z` |

Release Please updates package versions, dependency versions, manifest entries, and component changelogs. Do not manually create release tags or edit versions and changelogs during the normal release flow.

## 1. Merge Conventional Commits into `main`

Release Please determines the next version from Conventional Commits that affect each component path:

- `feat: add offline rescoring` produces a minor release.
- `fix: prevent a decoder crash` produces a patch release.
- `feat!: change the public decoder API` declares a breaking change.
- `docs:`, `test:`, and `chore:` commits normally do not produce a release by themselves.

Because this is a manifest repository, only components with relevant commits are released. A Rust library release does not imply a Flutter release, for example.

The `cargo-workspace` plugin links the two Rust packages. When `qwen-asr` changes, it updates the `qwen-asr` version used by `qwen-asr-cli` and may include the CLI in the same release PR with a dependency-only version bump.

## 2. Review the Release PR

Every push to `main` runs `.github/workflows/release-please.yml`. The workflow creates or updates the `release-please--branches--main` pull request, normally titled `chore: release main`.

Before merging it, verify:

1. The proposed versions match the intended release scope.
2. Each affected component's `CHANGELOG.md` is complete.
3. `crates/qwen-asr-cli/Cargo.toml` references the proposed `qwen-asr` version.
4. `.release-please-manifest.json`, `Cargo.toml` files, `Cargo.lock`, and Flutter `pubspec.yaml` are consistent.
5. Required tests and CI checks have passed.

Useful local preflight checks are:

```bash
cargo test --workspace
cargo publish --dry-run -p qwen-asr
cargo publish --dry-run -p qwen-asr-cli
```

For a Flutter release, also run:

```bash
cd flutter/qwen_asr
flutter pub get
flutter analyze
flutter test
dart pub publish --dry-run
```

## 3. Merge the Release PR

Merge the Release Please PR into `main`. The resulting push runs `.github/workflows/release-please.yml` again. Release Please then creates a GitHub tag and GitHub Release for each component included in the PR.

The `PAT` secret is deliberately used instead of the default `GITHUB_TOKEN`. Events created with the default token do not start follow-on workflows, while the CLI binary and Flutter publishing flows depend on tag or release events.

## 4. Automated Publishing

### Rust library: `qwen-asr`

When the Release Please action reports a `crates/qwen-asr` release, `.github/workflows/release-please.yml`:

1. Installs stable Rust and OpenBLAS on Ubuntu.
2. Runs `cargo publish` from `crates/qwen-asr` with `CARGO_REGISTRY_TOKEN`.
3. Publishes the version to crates.io.

### Rust CLI: `qwen-asr-cli`

When it reports a `crates/qwen-asr-cli` release, the same workflow:

1. Installs stable Rust and OpenBLAS.
2. Attempts to publish `qwen-asr` first, allowing an already-published version.
3. Waits 30 seconds for the crates.io index.
4. Publishes `qwen-asr-cli` to crates.io.

Creation of a `qwen-asr-cli-vX.Y.Z` GitHub Release also starts `.github/workflows/build-binaries.yml`. It builds, packages, and uploads these release assets:

- Apple Silicon macOS: `aarch64-apple-darwin`
- Linux: `x86_64-unknown-linux-gnu`

The archives are named `qwen-asr-X.Y.Z-TARGET.tar.gz` and contain the `qwen-asr` executable.

### Flutter plugin: `qwen_asr`

Creation of a `qwen_asr-vX.Y.Z` tag starts `.github/workflows/publish-flutter.yml`. It:

1. Installs stable Rust, Flutter, and Dart.
2. Runs `flutter pub get`.
3. Installs `flutter_rust_bridge_codegen` and generates the Rust bindings.
4. Uses pub.dev trusted publishing through OIDC.
5. Runs `dart pub publish --force` from `flutter/qwen_asr`.

## 5. Verify the Release

After merging the release PR, check the applicable workflows:

```bash
gh run list --workflow release-please.yml --limit 5
gh run list --workflow build-binaries.yml --limit 5
gh run list --workflow publish-flutter.yml --limit 5
```

Then verify all outputs included in the release:

- The expected tags and GitHub Releases exist.
- `qwen-asr` and/or `qwen-asr-cli` are visible on crates.io.
- A CLI GitHub Release contains both expected `.tar.gz` assets.
- A Flutter release is visible on pub.dev.

## Required Repository Configuration

- `PAT`: a fine-grained GitHub personal access token with Contents and Pull Requests read/write access. It lets Release Please create PRs, tags, and releases whose events can trigger the other workflows.
- `CARGO_REGISTRY_TOKEN`: a crates.io API token authorized to publish both Rust packages.
- pub.dev trusted publisher: configured for this GitHub repository and `.github/workflows/publish-flutter.yml`. Its tag pattern must be `qwen_asr-v{{version}}`.
- GitHub Actions permissions: the workflows require the `contents`, `pull-requests`, and `id-token` permissions declared in their YAML files.

## Failure Recovery

Inspect the failed step before retrying. Package registries do not allow republishing the same version, so first check whether the package reached crates.io or pub.dev even if the job ended in failure.

If the GitHub Release already exists but registry publishing failed, simply rerunning Release Please may skip publishing because the action no longer reports a newly created release. In that case, publish the already-versioned package manually with the appropriate registry credentials, or use a deliberately scoped recovery workflow. Do not create another tag for the same version.

If CLI binary building failed, rerun `.github/workflows/build-binaries.yml` for the existing CLI Release. If Flutter publishing failed, rerun the original tag-triggered `Publish to pub.dev` workflow after fixing its OIDC or build configuration.

## Workflow and Configuration Reference

- `.github/workflows/release-please.yml`: release PRs, GitHub Releases, and crates.io publishing.
- `.github/workflows/build-binaries.yml`: CLI binaries attached to CLI GitHub Releases.
- `.github/workflows/publish-flutter.yml`: OIDC publishing for Flutter tags.
- `release-please-config.json`: component paths, release types, and Cargo workspace integration.
- `.release-please-manifest.json`: most recently released version of each component.
