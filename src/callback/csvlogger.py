# callback/csv_logger.py

import csv
from src.callback.base import Callback

class CSVLogger(Callback):

    def __init__(self, filename):
        self.filename = filename
        self.header_written = False

    def on_epoch_end(self, model, epoch, logs):

        row = {"epoch": epoch}
        row.update(logs)

        with open(self.filename, "a", newline="") as f:

            writer = csv.DictWriter(
                f,
                fieldnames=row.keys()
            )

            if not self.header_written:
                writer.writeheader()
                self.header_written = True

            writer.writerow(row)