import wandb
from pathlib import Path


def wandb_create_run(run_name):
    if not run_name:
        return None
    run = wandb.init(
        project="Tralalero",
        name=run_name
    )
    return run


def wandb_finish_run(run):
    if run is not None:
        run.finish()


def wandb_log_code(run, code_dir : Path):
    if run is not None:
        run.log_code(code_dir.__str__())


def wandb_log_artifact(run, path_to_artifact : Path, artifact_type="artifact"):
    if run is None:
        return
    artifact = wandb.Artifact(name=path_to_artifact.name, type=artifact_type)
    artifact.add_file(path_to_artifact.__str__())
    run.log_artifact(artifact)


def wandb_load_artifact(run, artifact_full_name):
    if run is None or artifact_full_name is None:
        return None
    artifact = run.use_artifact(artifact_full_name, type='model')
    artifact_dir = artifact.download()
    path = list(Path(artifact_dir).resolve().iterdir())[0]
    return path
