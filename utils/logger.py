import os
import json
import logging
import pandas as pd

def set_logger(save=False, log_path=None):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if save and not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))

    if not logger.handlers:
        if save:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

def save_result_to_csv_and_json(items, path, anno=""):
    with open(os.path.join(path, '{}_result.json'.format(anno)), 'w+') as f:
        json.dump(items, f)
    
    df = pd.DataFrame(items)
    df.to_csv(os.path.join(path, '{}_result.csv'.format(anno)), index=False)
