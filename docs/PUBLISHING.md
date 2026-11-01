# Publishing the project safely

**Nothing is pushed to GitHub by any script.** These are commands for you to run, after you decide.

## Why not simply push the working repository?
Earlier phases committed `data/processed/corpus_processed.jsonl` and `queries_processed.jsonl` (full SciFact abstracts and claims) into the Git history. SciFact is **CC BY-NC 2.0**.
`.gitignore` now prevents *new* commits of those files, but it does not remove them from existing history, and executed notebooks/reports can contain local absolute paths.
`scripts/check_repo_hygiene.py` reports these.

## Option A (recommended): publish a clean export as a fresh repository
```powershell
python scripts/generate_readme_results.py            # make sure the README numbers match your reports
python scripts/generate_readme_results.py --check    # exit code 0 = up to date
python scripts/freeze_environment.py                 # writes requirements.lock.txt (exact versions you ran)
python scripts/check_repo_hygiene.py                 # review warnings about the working repo
python scripts/export_portfolio.py                   # creates dist\portfolio (refuses to overwrite; strict scan must pass)
```
1. **Choose a license for your code** and add a `LICENSE` file to the repository root *before* exporting (for example MIT or Apache-2.0; this is your decision and independent of the data/model licenses). Without one, others have no permission to reuse the code.
2. Review `dist\portfolio` (open `PUBLISH_MANIFEST.json`; confirm there are no dataset files, weights or personal paths).
3. Create the repository from the export:
```powershell
cd dist\portfolio
git init -b main
git config user.name "Your Name"
git config user.email "you@example.com"
git add -A
git status                                 # read the list; nothing from data\raw, artifacts\models, .env
git commit -m "Initial release: scientific semantic search (SciFact)"
```
4. On GitHub create an **empty** repository (no README, no license). Then, only when you are ready to publish:
```powershell
git remote add origin https://github.com/<your-user>/<your-repo>.git
git push -u origin main
```
Authenticate through the browser/Git Credential Manager (GitHub does not accept account passwords). **Never use `git push --force`** unless you are certain the remote holds nothing you want.

## Option B: keep the existing repository, but private
Pushing the existing history to a *private* repository is fine for backup. Do not make it public: the history contains dataset text.

## Option C: rewrite the existing history (advanced, destructive, not recommended)
Tools such as `git filter-repo` can remove the processed corpus from every commit. This rewrites all hashes and tags, so back up first (`git clone --mirror`) and do it only with a deliberate decision. Option A achieves the same outcome without rewriting anything.

## Checklist
- [ ] README numbers regenerated (`--check` passes)
- [ ] `check_repo_hygiene.py --mode export --root dist\portfolio` reports PASS
- [ ] LICENSE added; license/citation notes read ([DATA_AND_LICENSES.md](DATA_AND_LICENSES.md))
- [ ] No `.env`, no keys, no dataset text, no weights in the export
- [ ] Screenshots (if any) are real images taken from your own run, saved in `docs/screenshots/`
- [ ] You approved the push
