# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import os
import shutil
import sys
import tempfile

from . import USER_CONFIG_DIR
from .torch_utils import TORCH_1_9


def find_free_network_port() -> int:
    """
    Find a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer):
    """
    Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

    This function creates a temporary Python file that enables distributed training across multiple GPUs.
    The file contains the necessary configuration to initialize the trainer in a distributed environment.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing training configuration and arguments.
            Must have args attribute and be a class instance.

    Returns:
        (str): Path to the generated temporary DDP file.

    Notes:
        The generated file is saved in the USER_CONFIG_DIR/DDP directory and includes:
        - Trainer class import
        - Configuration overrides from the trainer arguments
        - Model path configuration
        - Training initialization code
    """
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    # Detect if overrides contain SSOD-specific keys to select the proper base cfg
    _ov = dict(vars(trainer.args))
    _ssod_keys = {
        "ssod", "ssod_weight", "batch_ssod", "mosaic_ssod", "mixup_ssod", "cutmix_ssod",
        "burn_in_epochs", "conf_threshold_high", "conf_threshold_low", "domain_adaptation",
        "pseudo_label_plots", "da/loss_s", "da/loss_t", "da/loss", "da_loss_weights",
        "use_loc_conf", "loc_conf_threshold", "two_axis_selection", "loc_conf_threshold_low",
        "loc_conf_threshold_high", "use_edge_conf", "edge_conf_threshold", "edge_conf_mask_mode",
        "edge_dfl_reweight", "edge_dfl_selector", "edge_dfl_weight", "edge_dfl_normalize",
        "oracle_edge_error_threshold",
        "assignment_stability_method", "assignment_perturbation"
    }
    use_ssod_cfg = any(k in _ov for k in _ssod_keys)

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {vars(trainer.args)}

if __name__ == "__main__":
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT, SSOD_DEFAULT_CFG_PATH

    # Choose base cfg depending on whether SSOD-specific keys are present
    _ov = dict(overrides)
    _ssod_keys = {list(_ssod_keys)}
    _use_ssod = any(k in _ov for k in _ssod_keys)
    if _use_ssod:
        cfg = str(SSOD_DEFAULT_CFG_PATH)
    else:
        cfg = DEFAULT_CFG_DICT.copy()
        cfg.update(save_dir='')
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
    results = trainer.train()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def generate_ddp_command(trainer):
    """
    Generate command for distributed training.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing configuration for distributed training.

    Returns:
        cmd (list[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/pytorch-lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{trainer.world_size}",
        "--master_port",
        f"{port}",
        file,
    ]
    return cmd, file


def ddp_cleanup(trainer, file):
    """
    Delete temporary file if created during distributed data parallel (DDP) training.

    This function checks if the provided file contains the trainer's ID in its name, indicating it was created
    as a temporary file for DDP training, and deletes it if so.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer used for distributed training.
        file (str): Path to the file that might need to be deleted.

    Examples:
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f"{id(trainer)}.py" in file:  # if temp_file suffix in file
        os.remove(file)
