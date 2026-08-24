# Publishing to GitHub

`release/iMED-pe_challenge/` is intended to be the root of the new standalone
repository. Do not push the parent challenge workspace or its Git history.

Keep the GitHub repository private until the challenge permits disclosure.
After reviewing the files, initialize and push with:

```bash
cd release/iMED-pe_challenge
git init -b main
git add .
git status
git commit -m "Initial reproducible iMED-PE methods release"
git remote add origin git@github.com:<account>/iMED-pe_challenge.git
git push -u origin main
```

If the GitHub repository was initialized with a README or license, clone that
empty repository first and copy this directory's contents into it instead of
force-pushing over the remote history.

Before the first public release:

```bash
make audit
make test
```

Then manually confirm that no staged file contains challenge data, generated
poses, private URLs, credentials, checkpoints, or local machine paths:

```bash
git diff --cached --stat
git diff --cached
```

Update `CITATION.cff` with the correct GitHub account and author names. Resolve
the provisional licensing notice before making the repository public.

