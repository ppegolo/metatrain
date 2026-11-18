# Upstream commits carried on `nep-gle`

This fork carries NEP and GLE work on top of upstream `metatrain`.  Anything
cherry-picked from upstream is recorded here so the eventual rebase can drop it
instead of re-resolving it.

## Cherry-picks

| local commit | upstream commit | subject |
| --- | --- | --- |
| `bb017d25` | `63abcae3` | Port augmentation to the batched `O3Transformations` API (upstream #1245) |

`63abcae3` was picked because the pinned `metatomic-torch` had already moved to
the batched `O3Transformations` API, so `utils/augmentation.py` on this branch
no longer ran against the installed dependency.  Files touched:

* `src/metatrain/utils/augmentation.py`
* `src/metatrain/utils/testing/output.py`
* `tests/utils/test_augmentation.py`
* `docs/src/dev-docs/changelog.rst`
* `pyproject.toml` — narrows `metatomic-torch` to `>=0.1.17,<0.2` and
  `metatomic-ase` to `>=0.1.1,<0.2`.  This pin is part of the picked commit, not
  a local decision, and it is the piece most likely to conflict on rebase.

**On rebase:** drop `bb017d25` once the branch is rebased onto an upstream
containing `63abcae3` — every hunk of it is already upstream.
