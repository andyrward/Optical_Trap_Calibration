# Issue-to-merge workflow

This repo uses GitHub coding agents for implementation. Opie, the analysis assistant, reviews each pull request before it is merged and checks the merged code on real data afterwards.

**Current rule (all PRs, until revisited):** Opie checks in with Andrew, and they discuss every pull request before anything is merged. Nothing is merged on Opie's review alone.

## Steps

| # | Step | Who |
|---|---|---|
| 1 | Write the issue: the problem, evidence, requested behavior, tests and acceptance criteria. | Opie drafts it, and Andrew reviews it |
| 2 | File the issue. | Andrew, or Opie with Andrew's OK |
| 3 | Assign the issue to a coding agent (for example, the Copilot coding agent). | Andrew, or Opie with Andrew's OK |
| 4 | Open a pull request (PR) that addresses the issue. | Coding agent |
| 5 | Run a Copilot code review on GitHub. | Andrew |
| 6 | **Pre-merge review:** check the PR against the issue's acceptance criteria and post the findings as a PR comment. | Opie |
| 7 | If the review fails, send it back to the agent with Opie's findings and repeat steps 4–6. | Andrew |
| 8 | Discuss the PR and Opie's review together, then merge only if both agree. | Andrew and Opie discuss, Andrew merges |
| 9 | **Post-merge check:** run the merged code on the tether1 files and report pass or fail. | Opie |
| 10 | File any new problems as new issues, starting again at step 1. | Opie drafts them, and Andrew reviews them |

## What Opie checks before merge
Opie checks out the PR branch locally and makes no pushes.
- **Fixes the issue.** Each acceptance criterion and each requested behavior is either met or listed as missing.
- **Tests pass.** The full suite runs, the tests the issue asked for are present, and none are skipped where real data is available.
- **No regressions.** Earlier fixes still hold. In particular:
  - `tether_id`, `tether_number` and `processing_id` are filled in on the rows, and provenance records the tether IDs and analysis parameters (the gap found in PR #16);
  - `--mode` is required, and unknown states are flagged rather than guessed;
  - there are no missing-value comparison crashes;
  - edge-correction output is unchanged unless the issue changes it.
- **Quick real-data run.** The changed command runs on at least one tether1 file without crashing, and its key numbers are compared with the expected values in the issue.
- **Result.** A PR comment that says pass or fail, lists each item with what was measured, and gives file and line numbers for any problem.

## What Opie verifies after merge
- Pull the merged code, run the full test suite and record the merge commit.
- Run the relevant commands on all four files, `tether1_1`, `tether1_2`, `tether1_final` and `tether1_overstretch`, with edge correction off and on.
- Report pass or fail for each acceptance criterion, with measured numbers next to the prototype or expected values where they can be compared. Reference values include:
  - Oscillator cycle counts of 5, 7, 10 and 0;
  - the first `tether1_1` ramp-down at 90.012–91.274 s;
  - the corrected `tether1_1` peak force of about 31 pN;
  - the five overstretch attempts, with their outcomes and peaks;
  - the 0.415 s relax curve.
- Note any new errors, warnings or unexpected states, and draft follow-up issues for them.

## GitHub access
All GitHub writes, such as filing issues, assigning agents and commenting on PRs, go through Andrew's fine-grained personal access token, logged in to the GitHub command-line tool (`gh`). They do not go through the GitHub connector app, which returns 403 for issue writes on this repo. The token needs Issues: Read and write; it also needs Pull requests: Read and write for PR comments, and Contents: Read and write only if pushing. Opie makes GitHub writes only with Andrew's OK, and never pushes to `main`.
