from contextlib import contextmanager
import builtins
import io
import os

@contextmanager
def deny_label_reads():
    opened = []
    blocked = []
    originals = [(builtins, 'open', builtins.open), (io, 'open', io.open), (os, 'open', os.open)]

    def wrap(fn):

        def guarded(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                name = os.fsdecode(path)
                if any((token in name for token in ('evaluation_labels', 'heldout_target_map', 'paired_target_map', 'target_gt'))):
                    blocked.append(name)
                    raise PermissionError('Evaluator labels forbidden during prediction')
                opened.append(name)
            return fn(path, *args, **kwargs)
        return guarded
    try:
        for module, key, fn in originals:
            setattr(module, key, wrap(fn))
        yield dict(opened_files=opened, blocked_label_attempts=blocked)
    finally:
        for module, key, fn in originals:
            setattr(module, key, fn)
