"""Small helpers for waiting on the physical start key."""


def wait_for_start_key(event):
    """Block until the start event is set, then consume it."""
    event.wait()
    event.clear()
