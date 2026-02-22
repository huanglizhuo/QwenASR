# QwenASR Automated Release Process

This guide explains how the automated release pipeline for `QwenASR` works and what you need to do to trigger a new release for both the Rust crate (`crates.io`) and the Flutter library (`pub.dev`).

This repository utilizes **[Release Please](https://github.com/googleapis/release-please)**. You do not need to manually edit `Cargo.toml`, `pubspec.yaml`, or write `CHANGELOG.md` files anymore.

## 🚀 How to Trigger a New Release

The release process relies entirely on **Conventional Commits** (e.g., `feat:`, `fix:`).

### Step 1: Write Code and Commit
When merging changes into the `main` branch, ensure your commits follow the conventional format:

* `feat: add offline rescoring` -> Triggers a **MINOR** release (e.g., `0.2.0` -> `0.3.0`)
* `fix: crash on audio initialization` -> Triggers a **PATCH** release (e.g., `0.2.1` -> `0.2.2`)
* `feat!: breaking API changes` -> Triggers a **MAJOR** release (e.g., `0.2.x` -> `1.0.0`)
* `chore/docs/test: update readme` -> Does **not** trigger a new release.

### Step 2: The Release Pull Request
1. Once your code is merged into `main`, GitHub Actions will silently create (or update) a Release PR titled `chore(main): release 0.2.x`.
2. This automated PR gathers all your `feat` and `fix` commits into a beautiful `CHANGELOG.md` and upgrades the version variables in your code automatically.
3. As long as you keep pushing code to `main`, the bot will keep updating this pending PR under the hood. It acts as an ongoing draft for your next release.

### Step 3: Approve and Publish
When you are ready to publish the packaged version:
1. Go to your Github [Pull Requests](https://github.com/huanglizhuo/RustQwenAsr/pulls).
2. Review the `chore(main): release...` PR.
3. Click **Squash and Merge** (or a standard merge) to bring it into `main`.

**That's it!** The moment you merge the PR, the following happens automatically:
* Release Please tags the repository using your Personal Access Token (PAT).
* This Tag triggers two completely automated workflows:
   * **crates.io Publishing**: Uploads the `qwen-asr` rust library.
   * **pub.dev Publishing**: Generates Rust bindings, authenticates via OIDC, and uploads the `qwen_asr` flutter package.

---

## 🛠️ Architecture & Workflows

If you ever need to debug the pipeline, here is how the GitHub Actions are structured:

1. **`.github/workflows/release-please.yml`**:
   * Listens to the `main` branch.
   * Runs the `release-please-action` bot using a custom GitHub `PAT` token.
   * Automatically publishes the `crates/qwen-asr` library to `crates.io` using the `CARGO_REGISTRY_TOKEN`.
   * **Does not publish Flutter**. It only tags the repo.

2. **`.github/workflows/publish-flutter.yml`**:
   * Listens **strictly for Flutter Tags** (e.g., `qwen_asr-vX.X.X`).
   * This decoupled pipeline ensures `pub.dev`'s strict OIDC security rules are fulfilled (pub.dev identity token exchange requires an explicit tag-based trigger, not a branch trigger).
   * It sets up Rust, installs `flutter_rust_bridge_codegen` via `cargo`, builds the bindings, and executes `dart pub publish` securely.

3. **`release-please-config.json`**:
   * Explicitly defines the paths (`crates/qwen-asr`, `flutter/qwen_asr`).
   * Ensures that tags are grouped intelligently and keeps the components separated.

## ⚠️ Important Configuration State

If you ever clone or copy this setup to a new repository, remember the hidden environment secrets that make this CI work:
* **`CARGO_REGISTRY_TOKEN`**: Repository Secret used to authorize crates.io publishes.
* **`PAT`**: Fine-grained Personal Access Token with read/write access to Pull Requests & Contents. It allows the Release Please bot to create tags that are capable of triggering secondary GitHub workflows (like `publish-flutter.yml`). Default `GITHUB_TOKEN` tags are banned from triggering loops.
* **Pub.dev OIDC Admin**: Configured on the pub.dev admin page for the package. **Tag Pattern MUST be `qwen_asr-v{{version}}`** for OIDC token exchanges to work correctly with this repository's tag formats.
