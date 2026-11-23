"""Read-only recording with real-sized control uploads; never sends robot commands."""
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time
from unittest.mock import patch

from .stream_recording_probe import run_probe
from .stream_storage import StreamStorage


def run_loaded_probe(source, seconds=45):
    source = Path(source).resolve()
    records = sorted(source.glob('command-*.json'))
    if not records:
        raise ValueError('Existing command records required as transport payload only')
    payload = max((p.read_bytes() for p in records), key=len)
    original = StreamStorage.start_readonly
    stop = threading.Event()
    threads, traffic, errors, stores = [], [], [], []

    def start(*args, **kwargs):
        storage = original(*args, **kwargs)
        stores.append(storage)
        def load():
            if stop.wait(7):
                return
            try:
                for index in range(8):
                    if stop.is_set():
                        return
                    # Only inert copies, outside the materializer's command namespace.
                    copies = 3 if index == 2 else 1
                    blobs = {f'load-evidence/burst-{index}-{n}.json': payload for n in range(copies)}
                    began = time.monotonic()
                    receipt = storage._push(blobs)
                    traffic.append({'index': index, 'files': copies, 'bytes': len(payload)*copies,
                                    'duration_s': time.monotonic()-began,
                                    'server_ack_verified': bool(receipt),
                                    'frame_status': storage.frame_spool.status()})
                    if stop.wait(2.5):
                        return
            except Exception as exc:
                errors.append(f'{type(exc).__name__}: {exc}')
        thread = threading.Thread(target=load, daemon=True)
        threads.append(thread)
        thread.start()
        return storage

    with patch.object(StreamStorage, 'start_readonly', side_effect=start):
        result = run_probe(seconds)
    stop.set()
    for thread in threads:
        thread.join(timeout=20)
    if any(thread.is_alive() for thread in threads):
        errors.append('Control upload thread did not finish')
    result['simultaneous_control_load'] = {
        'source': str(source), 'payload_sha256': hashlib.sha256(payload).hexdigest(),
        'payload_bytes': len(payload), 'bursts': traffic, 'errors': errors,
        'physical_commands_sent': 0,
        'passed': len(traffic) == 8 and not errors and result['status'] == 'measurement_complete'}
    if stores:
        path = stores[0].root / 'loaded-diagnostic-result.json'
        path.write_text(json.dumps(result, indent=2)+'\n')
        stores[0].sync_files(path)
        result['loaded_result_path'] = str(path)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=45)
    args = parser.parse_args()
    print(json.dumps(run_loaded_probe(args.source, args.seconds)), flush=True)
