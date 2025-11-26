"""
CLI: Model training + registry management

Commands:
    python -m src.cli.agent_train train
    python -m src.cli.agent_train register --name NAME --version VERSION --path PATH
    python -m src.cli.agent_train promote --name NAME --version VERSION
    python -m src.cli.agent_train list-models
"""

import argparse
import os
import json
from datetime import datetime

from src.deploy.model_registry import ModelRegistry
from src.learning.run_full_historical_training import main as run_training


# ------------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------------

def ensure_models_dir():
    if not os.path.exists("models"):
        os.makedirs("models")


# ------------------------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------------------------

def cmd_train(args):
    """
    Runs your full training pipeline:
        - numeric challenger
        - LLM challenger generations
        - strategy adapter
        - risk engine
        - observability
        - saves model + feature list
        - updates champion files
    """
    print("🏗 Running full model training pipeline...")
    result = run_training()  # this runs your entire Phase 1–7 system
    print("✅ Training complete.")

    #  result already writes models to /models and promotes champion automatically.
    return result


def cmd_register(args):
    """
    Adds an existing model into registry.
    """
    ensure_models_dir()
    reg = ModelRegistry()

    entry = reg.save(
        name=args.name,
        version=args.version,
        path=args.path,
        metadata={
            "registered_at": datetime.utcnow().isoformat(),
            "by": "user"
        }
    )

    print("✅ Model registered successfully:")
    print(json.dumps(entry, indent=2))


def cmd_promote(args):
    """
    Promote a model version to production.
    """
    reg = ModelRegistry()

    entry = reg.promote(
        name=args.name,
        version=args.version,
        by="user"
    )

    print("🎉 Model promoted to PRODUCTION:")
    print(json.dumps(entry, indent=2))


def cmd_list(args):
    """
    View all saved models.
    """
    reg = ModelRegistry()
    models = reg.list()

    print(json.dumps(models, indent=2))


# ------------------------------------------------------------------------------------
# CLI Entry Point
# ------------------------------------------------------------------------------------
def build_parser():
    parser = argparse.ArgumentParser(description="Model training / registry manager")

    sub = parser.add_subparsers(dest="cmd", required=True)

    # Train
    p_train = sub.add_parser("train", help="Run the full training pipeline")
    p_train.set_defaults(func=cmd_train)

    # Register
    p_register = sub.add_parser("register", help="Register an existing model")
    p_register.add_argument("--name", required=True)
    p_register.add_argument("--version", required=True)
    p_register.add_argument("--path", required=True)
    p_register.set_defaults(func=cmd_register)

    # Promote
    p_promote = sub.add_parser("promote", help="Promote a model version to PRODUCTION")
    p_promote.add_argument("--name", required=True)
    p_promote.add_argument("--version", required=True)
    p_promote.set_defaults(func=cmd_promote)

    # List models
    p_list = sub.add_parser("list-models", help="List all models in registry")
    p_list.set_defaults(func=cmd_list)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
