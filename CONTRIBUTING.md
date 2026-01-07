# Contributing

### License

Geti Prompt is licensed under the terms in [LICENSE](LICENSE). By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

### Sign your work

Please use the sign-off line at the end of the patch. Your signature certifies that you wrote the patch or otherwise have the right to pass it on as an open-source patch. The rules are pretty simple: if you can certify
the below (from [developercertificate.org](http://developercertificate.org/)):

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

Then you just add a line to every git commit message:

    Signed-off-by: Joe Smith <joe.smith@email.com>

Use your real name (sorry, no pseudonyms or anonymous contributions.)

If you set your `user.name` and `user.email` git configs, you can sign your
commit automatically with `git commit -s`.

### Suppressing False Positives

If necessary, to suppress _false_ positives, add inline comment with specific syntax.
Please also add a comment explaining _why_ you decided to disable a rule or provide a risk-acceptance reason.

#### Bandit

[Bandit](https://github.com/PyCQA/bandit) is a static analysis tool used to check Python code.
Findings can be ignored inline with `# nosec: BXXX` comments.

```python
import subprocess # nosec: B404 # this is actually fine
```

[Details](https://bandit.readthedocs.io/en/latest/config.html#exclusions) in Bandit docs.

#### Zizmor

[Zizmor](https://zizmor.sh/) is a static analysis tool used to check GitHub Actions workflows.
Findings can be ignored inline with `# zizmor: ignore[rulename]` comments.

```yaml
uses: actions/checkout@v3 # zizmor: ignore[artipacked] this is actually fine
```

[Details](https://docs.zizmor.sh/usage/#with-comments) in Zizmor docs.

#### Semgrep

Findings can be ignored inline with `# nosemgrep: rule-id` comments.

```python
    # nosemgrep: python.lang.security.audit.dangerous-system-call.dangerous-system-call # this is actually fine
    r = os.system(' '.join(command))
```

[Details](https://semgrep.dev/docs/ignoring-files-folders-code) in Semgrep docs.

### ChatOps

It is possible to trigger several workflows by commenting on pull requests. The following table summarizes these options.

| Comment  | Description                                               |
| -------- | --------------------------------------------------------- |
| `/build` | Triggers `distrib.yml` workflow to build container images |
