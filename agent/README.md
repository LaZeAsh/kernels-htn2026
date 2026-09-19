# The agent side

Nothing in this folder is submitted. It records experiment results and calls the
Dryft API. The engine source under `engine/` is submitted through the connected
GitHub repository. After pushing a revision, get the submission ID created by
the platform, then run that existing submission. The current live platform
accepts only `official` runs; a `public` request returns HTTP 422:

```sh
../bin/dryft run <submission-id> --mode official --wait 3000
```

`client.py` provides the run API through the Python standard library. It
requires `DRYFT_API` and `DRYFT_TOKEN`. `loop.py` takes an existing submission
ID, waits for the run, writes full details to `agent/runs/<run-id>.json`, and
appends the engine Git revision and source archive hash to
`agent/runs/history.jsonl`. It reports the raw geometric mean score in tokens
per second, and treats missing workload results, failed cases, or TTFT/TPOT
above 1.10 times native as failing latency gates. This local assessment is
useful even when the only runnable mode is `official`.

To resume observation of a run already started elsewhere, pass its run ID to
`loop.py`. This command only polls and records; it does not start a new run:

```sh
DRYFT_API=https://htn.dryft.ai python3 agent/loop.py <run-id> \
  --submission-id <submission-id>
```

`package.py` can still validate and hash an engine archive locally. The
platform no longer accepts direct archive uploads.
