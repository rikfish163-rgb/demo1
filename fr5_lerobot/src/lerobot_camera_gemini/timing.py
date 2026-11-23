"""Shared clock contracts; host monotonic time is comparable only on this host.

The camera endpoint is restricted to loopback. Raw Unix timestamps remain
audit data; they must not order frames/actions when monotonic stamps exist.
Legacy records without the new explicit field retain their Unix contract.
"""
import math

MONOTONIC = 'host_monotonic'
UNIX = 'host_unix'


def finite_time(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid positive source clock value')
    return float(value)


def camera_clock(meta):
    if 'source_timestamp_monotonic' in meta:
        return finite_time(meta['source_timestamp_monotonic']), MONOTONIC
    return finite_time(meta['source_timestamp_unix']), UNIX


def sample_clock(sample):
    captured,basis=camera_clock(sample['camera']['metadata'])
    if sample.get('sync_clock',basis) != basis:
        raise ValueError('Camera/state clock domain mismatch')
    key='received_monotonic' if basis==MONOTONIC else 'timestamp'
    return captured,finite_time(sample['robot_state'][key]),basis


def command_clock(command,basis):
    return finite_time(command['issued_monotonic' if basis==MONOTONIC else 'issued_unix'])


def completion_clock(row,basis):
    return finite_time(row['completed_monotonic' if basis==MONOTONIC else 'completed_at_unix'])
