# Remote SLURM Executor Plugin

## Installation

Use WSL on Windows.

First, install uv if you haven't already

```console
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Navigate to your project root and install the plugin:

```console
uv add git+https://github.com/lebrice/remote-slurm-executor
```

## Example

```python
from remote_slurm_executor import RemoteSlurmExecutor


def add(a, b):
    return a + b


cluster = "mila"
executor = RemoteSlurmExecutor(
    folder=f"logs/{cluster}/%j",
    cluster_hostname=cluster,
)
executor.update_parameters(partition="long")
# The submission interface is identical to concurrent.futures.Executor
job = executor.submit(add, 5, 7)  # will compute add(5, 7)
print(job.job_id)  # ID of your job

# waits for the submitted function to complete and returns its output
output = job.result()
# if ever the job failed, job.result() will raise an error with the corresponding trace
assert output == 12  # 5 + 7 = 12...  your addition was computed in the cluster
```
