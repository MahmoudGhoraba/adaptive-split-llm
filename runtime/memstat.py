"""
Memory measurements for the split runtime (macOS, MEASURED values).

  footprint      physical footprint of this process (what Activity Monitor shows as "Memory"); on Apple
                 unified memory it includes the process's GPU (MPS) buffers, unlike RSS
  peak_footprint lifetime maximum of the footprint
  rss            resident set size (includes clean file-backed pages, e.g. the mmapped safetensors file)
  mps_alloc      tensors currently allocated by PyTorch on MPS; mps_driver = what the MPS driver holds
  sys_available  memory the system can give without swapping; swap_used = system-wide swap in use

All values in bytes. snapshot() returns a dict; use gb() to print.
"""
import ctypes
import os

import psutil
import torch


class _RusageInfoV4(ctypes.Structure):          # struct rusage_info_v4, <sys/resource.h>
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [(n, ctypes.c_uint64) for n in (
        "user_time system_time pkg_idle_wkups interrupt_wkups pageins wired_size resident_size phys_footprint "
        "proc_start_abstime proc_exit_abstime child_user_time child_system_time child_pkg_idle_wkups "
        "child_interrupt_wkups child_pageins child_elapsed_abstime diskio_bytesread diskio_byteswritten "
        "cpu_time_qos_default cpu_time_qos_maintenance cpu_time_qos_background cpu_time_qos_utility "
        "cpu_time_qos_legacy cpu_time_qos_user_initiated cpu_time_qos_user_interactive billed_system_time "
        "serviced_system_time logical_writes lifetime_max_phys_footprint instructions cycles billed_energy "
        "serviced_energy interval_max_phys_footprint runnable_time").split()]


_libc = ctypes.CDLL("libSystem.dylib") if os.uname().sysname == "Darwin" else None


def _rusage():
    if _libc is None:
        return None
    r = _RusageInfoV4()
    if _libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(r)) != 0:   # RUSAGE_INFO_V4
        return None
    return r


def snapshot():
    r = _rusage()
    vm, sw = psutil.virtual_memory(), psutil.swap_memory()
    mps = torch.backends.mps.is_available()
    return dict(footprint=r.phys_footprint if r else None,
                peak_footprint=r.lifetime_max_phys_footprint if r else None,
                rss=psutil.Process().memory_info().rss,
                mps_alloc=torch.mps.current_allocated_memory() if mps else 0,
                mps_driver=torch.mps.driver_allocated_memory() if mps else 0,
                sys_available=vm.available, swap_used=sw.used)


def gb(s):
    return {k: (round(v / 1e9, 3) if isinstance(v, int) else v) for k, v in s.items()}


if __name__ == "__main__":                   # sanity check: a 0.5 GB MPS tensor must raise footprint by ~0.5 GB
    a = snapshot()
    x = torch.ones(250_000_000, dtype=torch.float16, device="mps"); torch.mps.synchronize()
    b = snapshot()
    print("before", gb(a)); print("after ", gb(b))
    print(f"footprint +{(b['footprint'] - a['footprint']) / 1e9:.3f} GB, rss +{(b['rss'] - a['rss']) / 1e9:.3f} GB "
          f"(expected footprint +0.500)")
