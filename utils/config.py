import json
from pathlib import Path

from model import build_network


FINAL_ARCHITECTURE = {
    "cnn_1": "densenet121",
    "cnn_2": "efficientnet_b0",
    "fusion": "gam",
    "classifier": "mlp",
}


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    required = {"dataset", "data_root", "run_name", "seed", "model", "training"}
    missing = sorted(required - set(config))
    if missing:
        raise ValueError("Missing config fields: " + ", ".join(missing))

    return config


def resolve_path(path, project_root):
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return Path(project_root) / path


def build_model_from_config(config, number_of_classes, device, pretrained=None):
    model_config = config["model"]

    for field, expected_value in FINAL_ARCHITECTURE.items():
        actual_value = model_config.get(field)
        if actual_value != expected_value:
            raise ValueError(
                f"This clean release supports {field}='{expected_value}', "
                f"but the config contains '{actual_value}'"
            )

    if pretrained is None:
        pretrained = model_config.get("pretrained", True)

    model = build_network(
        num_classes=number_of_classes,
        biformer_depth=model_config["biformer_depth"],
        patch_size=model_config["patch_size"],
        n_win=model_config["n_win"],
        topk=model_config["topk"],
        image_size=model_config["image_size"],
        use_pe=model_config["use_pe"],
        dropout=model_config["dropout"],
        mlp_dropout=model_config["mlp_dropout"],
        drop_path=model_config["drop_path"],
        pretrained=pretrained,
        cnn_1_freeze_blocks=model_config.get("cnn_1_freeze_blocks", 0),
        cnn_2_freeze_blocks=model_config.get("cnn_2_freeze_blocks", 0),
    )
    return model.to(device)


def validate_config(config):
    model = config["model"]
    training = config["training"]

    for field, expected_value in FINAL_ARCHITECTURE.items():
        if model.get(field) != expected_value:
            raise ValueError(f"model.{field} must be '{expected_value}'")

    image_size = model["image_size"]
    patch_size = model["patch_size"]
    n_win = model["n_win"]
    topk = model["topk"]

    if patch_size <= 0 or n_win <= 0:
        raise ValueError("patch_size and n_win must be greater than zero")
    if topk <= 0 or topk > n_win * n_win:
        raise ValueError("topk must be between 1 and n_win * n_win")
    if image_size % patch_size != 0:
        raise ValueError("image_size must be divisible by patch_size")
    if (image_size // patch_size) % n_win != 0:
        raise ValueError("The patch feature size must be divisible by n_win")
    if training["first_stage_epochs"] >= training["epochs"]:
        raise ValueError("first_stage_epochs must be smaller than epochs")
