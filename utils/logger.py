import csv
import os
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LOG_FILE = os.path.join(
    BASE_DIR,
    "logs",
    "detections.csv"
)


def save_alert(
    crop,
    stage,
    confidence,
    zone,
    risk,
    response
):
    os.makedirs(
        os.path.dirname(LOG_FILE),
        exist_ok=True
    )

    file_exists = os.path.exists(LOG_FILE)

    with open(LOG_FILE, "a", newline="") as file:

        writer = csv.writer(file)

        if not file_exists or os.path.getsize(LOG_FILE) == 0:

            writer.writerow([
                "Date",
                "Time",
                "Crop",
                "Stage",
                "Confidence",
                "Zone",
                "Risk",
                "Response"
            ])

        now = datetime.now()

        writer.writerow([
            now.strftime("%Y-%m-%d"),
            now.strftime("%H:%M:%S"),
            crop,
            stage,
            confidence,
            zone,
            risk,
            response
        ])