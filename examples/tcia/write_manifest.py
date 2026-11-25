"""Generate a CSV safely, including JSON Target maps."""
import csv
import json
from pathlib import Path

if __name__ == '__main__':
    output = Path('data/pairs.csv'); output.parent.mkdir(exist_ok=True)
    row = dict(pair_id='case-001', patient='subject-001', split='train',
        mr_image='mr/image.nrrd', us_image='us/image.nrrd',
        mr_prostate_label='mr/prostate_label.nrrd', us_prostate_label='us/prostate_label.nrrd',
        mr_targets=json.dumps({'1': 'mr/target_1_label.nrrd'}),
        us_targets=json.dumps({'1': 'us/target_1_label.nrrd'}))
    with output.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row)); writer.writeheader(); writer.writerow(row)

