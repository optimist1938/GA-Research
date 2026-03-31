import wandb
from pathlib import Path


def wandb_create_run(run_name):
    run = wandb.init(project="Tralalero", name=run_name or None)
    print(f"wandb run: {run.url}")
    return run


def wandb_finish_run(run):
    if run is not None:
        run.finish()


def wandb_log_code(run, code_dir: Path):
    if run is not None:
        run.log_code(str(code_dir))


def wandb_log_artifact(run, path_to_artifact: Path, artifact_type="artifact"):
    if run is None:
        return
    artifact = wandb.Artifact(name=path_to_artifact.name, type=artifact_type)
    artifact.add_file(str(path_to_artifact))
    run.log_artifact(artifact)
