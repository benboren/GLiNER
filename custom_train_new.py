# Standard library imports
import json
import os
import random

# Third-party imports
import numpy as np
import torch
from transformers import (
    TrainingArguments,
    set_seed
)
from gliner import GLiNER
from gliner.data_processing.collator import DataCollator
from gliner.modeling.loss_functions import span_model_custom_loss
from swarm_one.hugging_face import Client

# Constants
SEED = 42
TRAIN_PATH = "resultado-treino.json"
MODEL_CHECKPOINT = "urchade/gliner_small"
OUTPUT_DIR = "models-adjust"
NUM_STEPS = 500
BATCH_SIZE = 8
TRAIN_TEST_SPLIT = 0.9


def set_random_seeds(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_and_split_data(file_path: str, split_ratio: float):
    """Load and split data into train and test sets."""
    with open(file_path, "r") as f:
        data = json.load(f)

    print(f'Dataset size: {len(data)}')
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)

    return data[:split_idx], data[split_idx:]


def get_training_args(num_epochs: int, batch_size: int, num_batches: int):
    """Configure training arguments."""
    return TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        optim="adamw_torch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        evaluation_strategy="steps",
        save_steps=100,
        max_steps=num_batches,
        save_total_limit=10,
        dataloader_num_workers=0,
        use_cpu=False,
        report_to="none",
        remove_unused_columns=False
    )


def main():
    # Initialize environment and seeds
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    set_random_seeds(SEED)

    # Initialize SwarmOne client
    swarm_one_client = Client(api_key='API_KEY')

    # Load and prepare data
    train_dataset, test_dataset = load_and_split_data(TRAIN_PATH, TRAIN_TEST_SPLIT)
    print('Dataset is split...')

    # Initialize model and data collator
    model = GLiNER.from_pretrained(MODEL_CHECKPOINT)
    data_collator = DataCollator(
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True
    )

    # Calculate training parameters
    data_size = len(train_dataset)
    num_batches = data_size // BATCH_SIZE
    num_epochs = max(1, NUM_STEPS // num_batches)

    # Configure training arguments
    training_args = get_training_args(num_epochs, BATCH_SIZE, num_batches)

    # Start training
    job_id = swarm_one_client.fit(
        # trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
        compute_loss_func=span_model_custom_loss
    )

    job_history = swarm_one_client.get_job_history(job_id)
    job_tasks_ids = swarm_one_client.get_job_tasks_ids(job_id)
    task_id = job_tasks_ids[0]
    # Load trained model
    trained_state_dict = swarm_one_client.download_trained_state_dict(task_id=task_id)
    model.state_dict = trained_state_dict
    return model


if __name__ == "__main__":
    main()
