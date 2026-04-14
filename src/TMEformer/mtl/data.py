import os
from .collators import DataCollatorForMultitaskCellClassification
from .imports import *

def validate_columns(dataset, required_columns, dataset_type):
    """Ensures required columns are present in the dataset."""
    missing_columns = [col for col in required_columns if col not in dataset.column_names]
    if missing_columns:
        raise KeyError(
            f"Missing columns in {dataset_type} dataset: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )


def create_label_mappings(dataset, task_to_column):
    """Creates label mappings for the dataset."""
    task_label_mappings = {}
    num_labels_list = []
    for task, column in task_to_column.items():
        unique_values = sorted(set(dataset[column]))
        mapping = {label: idx for idx, label in enumerate(unique_values)}
        task_label_mappings[task] = mapping
        num_labels_list.append(len(unique_values))
    return task_label_mappings, num_labels_list


def save_label_mappings(mappings, path):
    """Saves label mappings to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(mappings, f)


def load_label_mappings(path):
    """Loads label mappings from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def transform_dataset(dataset, task_to_column, task_label_mappings, config, is_test):
    """Transforms the dataset to the required format."""
    transformed_dataset = []
    cell_id_mapping = {}

    for idx, record in enumerate(dataset):
        transformed_record = {
            "input_ids": torch.tensor(record["input_ids"], dtype=torch.long),
            "cell_id": idx,  # Index-based cell ID
        }

        if not is_test:
            label_dict = {
                task: task_label_mappings[task][record[column]]
                for task, column in task_to_column.items()
            }
        else:
            label_dict = {task: -1 for task in config["task_names"]}

        transformed_record["label"] = label_dict
        transformed_dataset.append(transformed_record)
        cell_id_mapping[idx] = record.get("unique_cell_id", idx)

    return transformed_dataset, cell_id_mapping


def load_and_preprocess_data(dataset_path, config, is_test=False, dataset_type=""):
    """Main function to load and preprocess data."""
    try:
        dataset = load_from_disk(dataset_path)

        # Setup task and column mappings
        task_names = [f"task{i+1}" for i in range(len(config["task_columns"]))]
        task_to_column = dict(zip(task_names, config["task_columns"]))
        config["task_names"] = task_names

        label_mappings_path = os.path.join(
            config["results_dir"],
            f"task_label_mappings{'_val' if dataset_type == 'validation' else ''}.pkl"
        )

        if not is_test:
            validate_columns(dataset, task_to_column.values(), dataset_type)

            # Create and save label mappings
            task_label_mappings, num_labels_list = create_label_mappings(dataset, task_to_column)
            save_label_mappings(task_label_mappings, label_mappings_path)
        else:
            # Load existing mappings for test data
            task_label_mappings = load_label_mappings(label_mappings_path)
            num_labels_list = [len(mapping) for mapping in task_label_mappings.values()]

        # Transform dataset
        transformed_dataset, cell_id_mapping = transform_dataset(
            dataset, task_to_column, task_label_mappings, config, is_test
        )

        return transformed_dataset, cell_id_mapping, num_labels_list

    except KeyError as e:
        raise ValueError(f"Configuration error or dataset key missing: {e}")
    except Exception as e:
        raise RuntimeError(f"Error during data loading or preprocessing: {e}")


def preload_and_process_data(config):
    """Preloads and preprocesses train and validation datasets."""
    # Process train data and save mappings
    train_data = load_and_preprocess_data(config["train_path"], config, dataset_type="train")

    # Process validation data and save mappings
    val_data = load_and_preprocess_data(config["val_path"], config, dataset_type="validation")

    # Validate that the mappings match
    validate_label_mappings(config)

    return (*train_data, *val_data[:2])  # Return train and val data along with mappings


def validate_label_mappings(config):
    """Ensures train and validation label mappings are consistent."""
    train_mappings_path = os.path.join(config["results_dir"], "task_label_mappings.pkl")
    val_mappings_path = os.path.join(config["results_dir"], "task_label_mappings_val.pkl")
    train_mappings = load_label_mappings(train_mappings_path)
    val_mappings = load_label_mappings(val_mappings_path)

    for task_name in config["task_names"]:
        if train_mappings[task_name] != val_mappings[task_name]:
            raise ValueError(
                f"Mismatch in label mappings for task '{task_name}'.\n"
                f"Train Mapping: {train_mappings[task_name]}\n"
                f"Validation Mapping: {val_mappings[task_name]}"
            )


def get_data_loader(preprocessed_dataset, batch_size):
    """Creates a DataLoader with optimal settings."""
    return DataLoader(
        preprocessed_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=DataCollatorForMultitaskCellClassification(),
        num_workers=os.cpu_count(),
        pin_memory=True,
    )


def preload_data(config):
    """Preprocesses train and validation data for trials."""
    train_loader = get_data_loader(*preload_and_process_data(config)[:2], config["batch_size"])
    val_loader = get_data_loader(*preload_and_process_data(config)[2:4], config["batch_size"])
    return train_loader, val_loader


def load_and_preprocess_test_data(config):
    """Loads and preprocesses test data."""
    return load_and_preprocess_data(config["test_path"], config, is_test=True)


def prepare_test_loader(config):
    """Prepares DataLoader for test data."""
    test_dataset, cell_id_mapping, num_labels_list = load_and_preprocess_test_data(config)
    test_loader = get_data_loader(test_dataset, config["batch_size"])
    return test_loader, cell_id_mapping, num_labels_list
