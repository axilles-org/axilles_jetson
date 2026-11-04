"""Expose the archived November 2 CSV in the dashboard, without replacing its DB."""
from pathlib import Path
import hashlib
import json
import sqlite3
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'ML_model/runs/imported/backup_ml_control_model_4745a55'
RUN_ID = '20261102_182941_1d037a'


def main():
    csv = SOURCE / 'ML_model/logs/powered_run.csv'
    df = pd.read_csv(csv)
    with sqlite3.connect(SOURCE / 'dashboard/data/runs.db') as source:
        source.row_factory = sqlite3.Row
        matches = source.execute('SELECT * FROM runs WHERE sample_count=?', (len(df),)).fetchall()
        assert len(matches) == 1 and matches[0]['run_id'] == RUN_ID
        record = dict(matches[0])
    provenance = dict(source_commit='4745a55', source_csv='ML_model/logs/powered_run.csv',
                      sha256=hashlib.sha256(csv.read_bytes()).hexdigest(),
                      association='Inferred from unique sample-count match and latest source DB record; CSV has no run ID.',
                      conversion='CSV columns and values preserved. No absolute or monotonic timestamps invented.')
    meta = json.loads(record['meta_json'])
    meta['import_provenance'] = provenance
    record['meta_json'] = json.dumps(meta)
    record['notes'] = 'Imported CSV from backup/ml_control_model; run association inferred (see metadata). IMUs frozen during assistance.'
    directory = ROOT / 'dashboard/data/runs' / RUN_ID
    directory.mkdir(parents=True, exist_ok=True)
    part = directory / 'part-00000.parquet'
    if part.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(part), df)
    else:
        df.to_parquet(part, index=False)
    (directory / 'import_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')
    with sqlite3.connect(ROOT / 'dashboard/data/runs.db') as target:
        backup_path = SOURCE / 'local_runs_before_import.db'
        if not backup_path.exists():
            with sqlite3.connect(backup_path) as backup:
                target.backup(backup)
        keys = list(record)
        # An existing local run is never replaced, including its notes or metadata.
        target.execute(f"INSERT OR IGNORE INTO runs ({','.join(keys)}) VALUES ({','.join('?' for _ in keys)})", list(record.values()))
    pd.testing.assert_frame_equal(pd.read_parquet(part), df)
    print(f'Imported {len(df)} rows into {directory}; older local runs preserved.')


if __name__ == '__main__':
    main()
