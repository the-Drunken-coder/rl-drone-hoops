#!/usr/bin/env python3
"""
hoopsctl.py - Ops helper for rl-drone-hoops.

Humans:
  - Run with no args for an arrow-key menu UI.

Agents:
  - Use subcommands (start/stop/status/list/logs/eval/purge/doctor).
  - Add --json for machine-readable output.

This script intentionally avoids third-party dependencies.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / 'runs'
VIDEOS_DIR = ROOT / 'videos'


def _is_windows() -> bool:
    return os.name == 'nt'


def _is_posix() -> bool:
    return os.name == 'posix'


def _now_ts() -> str:
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def _die(msg: str, *, code: int = 2) -> 'NoReturn':
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def _run(cmd: Sequence[str], *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(list(cmd), check=check, capture_output=capture, text=True)


def _read_text(p: Path) -> str:
    return p.read_text(encoding='utf-8', errors='replace')


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding='utf-8')


def _read_json(p: Path) -> Dict[str, Any]:
    return json.loads(_read_text(p))


def _write_json(p: Path, obj: Dict[str, Any]) -> None:
    _write_text(p, json.dumps(obj, indent=2, sort_keys=True) + '\n')


def _human_bytes(n: int) -> str:
    units = ['B', 'KB', 'MB', 'GB', 'TB']
    x = float(n)
    for u in units:
        if x < 1024.0 or u == units[-1]:
            if u == 'B':
                return f"{int(x)} {u}"
            return f"{x:.1f} {u}"
        x /= 1024.0
    return f"{n} B"


def _dir_size_bytes(p: Path) -> int:
    total = 0
    if not p.exists():
        return 0
    for root, _dirs, files in os.walk(p):
        for fn in files:
            try:
                total += (Path(root) / fn).stat().st_size
            except OSError:
                # Ignore files that cannot be accessed when computing directory size.
                pass
    return total


@dataclass(frozen=True)
class RunInfo:
    name: str
    run_dir: Path
    meta_path: Path
    meta: Dict[str, Any]


def _run_meta_path(run_dir: Path) -> Path:
    return run_dir / 'hoopsctl.json'


def _load_run_info(run_dir: Path) -> Optional[RunInfo]:
    mp = _run_meta_path(run_dir)
    if not mp.exists():
        return None
    try:
        meta = _read_json(mp)
    except Exception:
        return None
    return RunInfo(name=run_dir.name, run_dir=run_dir, meta_path=mp, meta=meta)


def list_run_dirs(*, include_discarded: bool = False) -> List[Path]:
    if not RUNS_DIR.exists():
        return []
    out: List[Path] = []
    for p in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        if not include_discarded and p.name.startswith('_'):
            continue
        out.append(p)
    return out


def _default_run_name() -> str:
    return f"ppo_rnn_{_now_ts()}"


def _python_default() -> str:
    return 'python3' if _is_posix() else 'python'


def _ensure_ansi_on_windows() -> None:
    if not _is_windows():
        return
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
        GetStdHandle = kernel32.GetStdHandle
        GetConsoleMode = kernel32.GetConsoleMode
        SetConsoleMode = kernel32.SetConsoleMode
        GetStdHandle.argtypes = [wintypes.DWORD]
        GetStdHandle.restype = wintypes.HANDLE
        GetConsoleMode.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        GetConsoleMode.restype = wintypes.BOOL
        SetConsoleMode.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        SetConsoleMode.restype = wintypes.BOOL

        STD_OUTPUT_HANDLE = wintypes.DWORD(-11)  # type: ignore[arg-type]
        h = GetStdHandle(STD_OUTPUT_HANDLE)
        mode = wintypes.DWORD()
        if not GetConsoleMode(h, ctypes.byref(mode)):
            return
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        new_mode = wintypes.DWORD(mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)
        SetConsoleMode(h, new_mode)
    except Exception:
        return


def _tmux_available() -> bool:
    return _which('tmux') is not None


def _wsl_available() -> bool:
    return _is_windows() and _which('wsl.exe') is not None


def shlex_quote(s: str) -> str:
    if s == '':
        return "''"
    safe = "@%_+=:,./-"
    if all(c.isalnum() or c in safe for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _wsl_exec(cmd: str, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(['wsl.exe', '-e', 'bash', '-lc', cmd], check=check, capture_output=capture, text=True)


def _wslpath_u(win_path: Path) -> str:
    cp = subprocess.run(['wsl.exe', '-e', 'bash', '-lc', f"wslpath -a -u {shlex_quote(str(win_path))}"], check=True, capture_output=True, text=True)
    return (cp.stdout or '').strip()


@dataclass
class StartOpts:
    run_name: str
    run_dir: Path
    python: str
    total_steps: int
    num_envs: int
    extra_args: List[str]
    resume: bool
    checkpoint: str
    attach: bool
    json_out: bool


def _start_detached_process(cmd: Sequence[str], *, train_log: Path, env_extra: Dict[str, str]) -> subprocess.Popen:
    train_log.parent.mkdir(parents=True, exist_ok=True)
    f = open(train_log, 'ab', buffering=0)
    env = os.environ.copy()
    # Reduce thread oversubscription for more predictable CPU usage.
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    env.update(env_extra)
    try:
        if _is_windows():
            creationflags = 0x00000200 | 0x00000008  # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS
            proc = subprocess.Popen(list(cmd), cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=creationflags)
        else:
            proc = subprocess.Popen(list(cmd), cwd=str(ROOT), env=env, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, preexec_fn=os.setsid)  # type: ignore[arg-type]
        return proc
    finally:
        # Close file handle in parent process; child still has its own handle
        f.close()


def start_training(opts: StartOpts) -> Dict[str, Any]:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    opts.run_dir.mkdir(parents=True, exist_ok=True)
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)

    train_log = opts.run_dir / 'train.log'
    py_path = str(ROOT)
    existing_py_path = os.environ.get('PYTHONPATH')
    if existing_py_path:
        py_path = py_path + os.pathsep + existing_py_path

    base_cmd: List[str] = [opts.python, '-u', str((ROOT / 'scripts' / 'train_recurrent_ppo.py').resolve()), '--run-dir', str(opts.run_dir), '--total-steps', str(int(opts.total_steps)), '--num-envs', str(int(opts.num_envs))]
    if opts.checkpoint:
        base_cmd += ['--checkpoint', opts.checkpoint]
    elif opts.resume:
        base_cmd += ['--resume']
    if opts.extra_args:
        base_cmd += list(opts.extra_args)

    meta: Dict[str, Any] = {'run_name': opts.run_name, 'run_dir': str(opts.run_dir), 'created_at': datetime.now().isoformat(timespec='seconds'), 'platform': {'os': os.name, 'platform': platform.platform()}, 'video_dir': str(VIDEOS_DIR.resolve()), 'mode': '', 'pid': None, 'tmux_session': None, 'cmd': base_cmd}

    if _is_posix():
        if _tmux_available():
            session = opts.run_name
            cmd_str = f"export RL_DRONE_HOOPS_VIDEO_DIR={shlex_quote(str(VIDEOS_DIR.resolve()))}; export PYTHONPATH={shlex_quote(py_path)}; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; mkdir -p {shlex_quote(str(opts.run_dir))}; {' '.join(shlex_quote(x) for x in base_cmd)} 2>&1 | tee -a {shlex_quote(str(train_log))}"
            _run(['tmux', 'new-session', '-d', '-s', session, 'bash', '-lc', cmd_str], check=True)
            meta['mode'] = 'tmux'
            meta['tmux_session'] = session
            _write_json(_run_meta_path(opts.run_dir), meta)
            if opts.attach:
                os.execvp('tmux', ['tmux', 'attach', '-t', session])
            return {'ok': True, 'mode': 'tmux', 'run_name': opts.run_name, 'run_dir': str(opts.run_dir), 'tmux_session': session}

        proc = _start_detached_process(base_cmd, train_log=train_log, env_extra={'RL_DRONE_HOOPS_VIDEO_DIR': str(VIDEOS_DIR.resolve()), 'PYTHONPATH': py_path})
        meta['mode'] = 'detached_no_tmux'
        meta['pid'] = proc.pid
        _write_json(_run_meta_path(opts.run_dir), meta)
        return {'ok': True, 'mode': meta['mode'], 'run_name': opts.run_name, 'run_dir': str(opts.run_dir), 'pid': proc.pid}

    if _wsl_available():
        have_tmux = False
        try:
            _wsl_exec('command -v tmux >/dev/null 2>&1', check=True)
            have_tmux = True
        except Exception:
            have_tmux = False

        if have_tmux:
            session = opts.run_name
            root_u = _wslpath_u(ROOT)
            run_dir_u = _wslpath_u(opts.run_dir)
            videos_u = _wslpath_u(VIDEOS_DIR.resolve())
            train_log_u = _wslpath_u(train_log)

            base_cmd_u = ['python3', '-u', f"{root_u}/scripts/train_recurrent_ppo.py", '--run-dir', run_dir_u, '--total-steps', str(int(opts.total_steps)), '--num-envs', str(int(opts.num_envs))]
            if opts.checkpoint:
                base_cmd_u += ['--checkpoint', opts.checkpoint]
            elif opts.resume:
                base_cmd_u += ['--resume']
            if opts.extra_args:
                base_cmd_u += list(opts.extra_args)

            cmd_str = f"cd {shlex_quote(root_u)}; export RL_DRONE_HOOPS_VIDEO_DIR={shlex_quote(videos_u)}; export PYTHONPATH={shlex_quote(root_u)}; export OMP_NUM_THREADS=1; export MKL_NUM_THREADS=1; mkdir -p {shlex_quote(run_dir_u)}; {' '.join(shlex_quote(x) for x in base_cmd_u)} 2>&1 | tee -a {shlex_quote(train_log_u)}"
            _wsl_exec(f"tmux new-session -d -s {shlex_quote(session)} bash -lc {shlex_quote(cmd_str)}", check=True)
            meta['mode'] = 'wsl_tmux'
            meta['tmux_session'] = session
            _write_json(_run_meta_path(opts.run_dir), meta)
            return {'ok': True, 'mode': 'wsl_tmux', 'run_name': opts.run_name, 'run_dir': str(opts.run_dir), 'tmux_session': session}

    proc = _start_detached_process(base_cmd, train_log=train_log, env_extra={'RL_DRONE_HOOPS_VIDEO_DIR': str(VIDEOS_DIR.resolve()), 'PYTHONPATH': py_path})
    meta['mode'] = 'win_detached'
    meta['pid'] = proc.pid
    _write_json(_run_meta_path(opts.run_dir), meta)
    return {'ok': True, 'mode': meta['mode'], 'run_name': opts.run_name, 'run_dir': str(opts.run_dir), 'pid': proc.pid}


def _tmux_list_sessions() -> List[str]:
    if not _tmux_available():
        return []
    cp = _run(['tmux', 'list-sessions', '-F', '#{session_name}'], check=False, capture=True)
    if cp.returncode != 0:
        return []
    return [ln.strip() for ln in (cp.stdout or '').splitlines() if ln.strip()]


def _wsl_tmux_list_sessions() -> List[str]:
    if not _wsl_available():
        return []
    try:
        cp = _wsl_exec("tmux list-sessions -F '#{session_name}'", check=False, capture=True)
        if cp.returncode != 0:
            return []
        return [ln.strip() for ln in (cp.stdout or '').splitlines() if ln.strip()]
    except Exception:
        return []


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if _is_windows():
        cp = subprocess.run(['tasklist', '/FI', f'PID eq {pid}'], capture_output=True, text=True)
        return str(pid) in (cp.stdout or '')
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _stop_pid(pid: int) -> bool:
    if pid <= 0:
        return False
    if _is_windows():
        cp = subprocess.run(['taskkill', '/PID', str(pid), '/T', '/F'], capture_output=True, text=True)
        return cp.returncode == 0
    try:
        os.killpg(pid, signal.SIGTERM)
        return True
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False


def stop_run(*, run_name: str, json_out: bool = False) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_name
    info = _load_run_info(run_dir)
    if info and info.meta.get('mode') in ('tmux', 'wsl_tmux'):
        sess = str(info.meta.get('tmux_session') or run_name)
        if info.meta.get('mode') == 'tmux':
            _run(['tmux', 'kill-session', '-t', sess], check=False)
            return {'ok': True, 'stopped': True, 'mode': 'tmux', 'tmux_session': sess, 'run_name': run_name}
        _wsl_exec(f"tmux kill-session -t {shlex_quote(sess)}", check=False)
        return {'ok': True, 'stopped': True, 'mode': 'wsl_tmux', 'tmux_session': sess, 'run_name': run_name}

    pid = None
    if info:
        try:
            pid = int(info.meta.get('pid') or 0)
        except Exception:
            pid = None
    if pid:
        alive = _pid_alive(pid)
        stopped = _stop_pid(pid)
        return {'ok': True, 'run_name': run_name, 'mode': str((info.meta.get('mode') if info else '') or 'pid'), 'pid': pid, 'was_alive': alive, 'stopped': stopped}

    if _is_posix() and _tmux_available():
        _run(['tmux', 'kill-session', '-t', run_name], check=False)
        return {'ok': True, 'run_name': run_name, 'mode': 'tmux_guess', 'stopped': True}
    if _is_windows() and _wsl_available():
        _wsl_exec(f"tmux kill-session -t {shlex_quote(run_name)}", check=False)
        return {'ok': True, 'run_name': run_name, 'mode': 'wsl_tmux_guess', 'stopped': True}

    return {'ok': False, 'error': f"Could not determine how to stop run '{run_name}' (no hoopsctl.json, no pid)."}


def status(*, json_out: bool = False) -> Dict[str, Any]:
    tmux = _tmux_list_sessions() if _is_posix() else []
    wsl_tmux = _wsl_tmux_list_sessions() if _is_windows() else []

    managed: List[Dict[str, Any]] = []
    for rd in list_run_dirs(include_discarded=False):
        info = _load_run_info(rd)
        if not info:
            continue
        meta = dict(info.meta)
        mode = str(meta.get('mode') or '')
        alive = None
        if mode in ('tmux',):
            alive = str(meta.get('tmux_session') or info.name) in tmux
        elif mode in ('wsl_tmux',):
            alive = str(meta.get('tmux_session') or info.name) in wsl_tmux
        else:
            try:
                pid = int(meta.get('pid') or 0)
            except Exception:
                pid = 0
            alive = _pid_alive(pid) if pid else False
        managed.append({'run_name': info.name, 'mode': mode, 'alive': bool(alive), 'run_dir': str(rd), 'pid': meta.get('pid'), 'tmux_session': meta.get('tmux_session')})

    return {'ok': True, 'tmux_sessions': tmux, 'wsl_tmux_sessions': wsl_tmux, 'managed_runs': managed}


def _tail_file(path: Path, n: int = 50) -> str:
    if not path.exists():
        return ''
    try:
        with path.open('rb') as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            read_n = min(size, 64 * 1024)
            f.seek(-read_n, os.SEEK_END)
            data = f.read().decode('utf-8', errors='replace')
        lines = data.splitlines()
        return '\n'.join(lines[-n:]) + ('\n' if lines else '')
    except Exception:
        return ''


def show_logs(*, run_name: str, n: int = 60) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_name
    log_path = run_dir / 'train.log'
    tail = _tail_file(log_path, n=n)
    return {'ok': True, 'run_name': run_name, 'log_path': str(log_path), 'tail': tail}


def list_runs(*, json_out: bool = False) -> Dict[str, Any]:
    runs: List[Dict[str, Any]] = []
    for rd in list_run_dirs(include_discarded=False):
        ckpts = rd / 'checkpoints'
        tb = rd / 'tb'
        runs.append({'run_name': rd.name, 'run_dir': str(rd), 'has_log': (rd / 'train.log').exists(), 'checkpoints': len(list(ckpts.glob('step*.pt'))) if ckpts.exists() else 0, 'tb_dir': str(tb) if tb.exists() else '', 'size': _human_bytes(_dir_size_bytes(rd)), 'managed': (_run_meta_path(rd).exists())})
    return {'ok': True, 'runs': runs}


def eval_latest(*, run_name: str, episodes: int = 3, video: bool = True, json_out: bool = False) -> Dict[str, Any]:
    run_dir = RUNS_DIR / run_name
    ckpt_dir = run_dir / 'checkpoints'
    if not ckpt_dir.exists():
        return {'ok': False, 'error': f'No checkpoints dir: {ckpt_dir}'}
    ckpts = sorted([p for p in ckpt_dir.glob('step*.pt') if p.is_file()], reverse=True)
    if not ckpts:
        return {'ok': False, 'error': f'No checkpoints found under {ckpt_dir}'}
    ckpt = ckpts[0]
    py = _python_default()
    cmd = [py, '-u', str((ROOT / 'scripts' / 'eval_checkpoint.py').resolve()), str(ckpt), '--episodes', str(int(episodes))]
    if video:
        cmd += ['--video', '--video-dir', str(VIDEOS_DIR.resolve())]
    cp = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    return {'ok': cp.returncode == 0, 'run_name': run_name, 'checkpoint': str(ckpt), 'cmd': cmd, 'returncode': cp.returncode, 'stdout': cp.stdout, 'stderr': cp.stderr}


def purge_results(*, purge_runs: bool, purge_videos: bool, yes: bool) -> Dict[str, Any]:
    if not yes:
        return {'ok': False, 'error': 'Refusing to delete without --yes.'}
    deleted: List[str] = []
    errors: List[str] = []
    if purge_runs and RUNS_DIR.exists():
        for p in RUNS_DIR.iterdir():
            if not p.is_dir():
                continue
            try:
                shutil.rmtree(p)
                deleted.append(str(p))
            except Exception as e:
                errors.append(f'{p}: {e}')
    if purge_videos and VIDEOS_DIR.exists():
        for p in VIDEOS_DIR.iterdir():
            try:
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                deleted.append(str(p))
            except Exception as e:
                errors.append(f'{p}: {e}')
    return {'ok': not errors, 'deleted': deleted, 'errors': errors}


def doctor(*, json_out: bool = False) -> Dict[str, Any]:
    info: Dict[str, Any] = {'ok': True, 'python': sys.version.replace('\n', ' '), 'platform': platform.platform(), 'root': str(ROOT), 'runs_dir_exists': RUNS_DIR.exists(), 'videos_dir_exists': VIDEOS_DIR.exists(), 'tmux': bool(_which('tmux')), 'wsl': bool(_which('wsl.exe')) if _is_windows() else False}
    for mod in ['mujoco', 'gymnasium', 'numpy', 'torch']:
        try:
            __import__(mod)
            info[f'import_{mod}'] = True
        except Exception as e:
            info[f'import_{mod}'] = False
            info[f'import_{mod}_err'] = str(e)
    return info


class _Keys:
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'
    ENTER = 'enter'
    ESC = 'esc'
    QUIT = 'quit'
    OTHER = 'other'


def _read_key() -> str:
    if _is_windows():
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ('\x00', '\xe0'):
            ch2 = msvcrt.getwch()
            return {'H': _Keys.UP, 'P': _Keys.DOWN, 'K': _Keys.LEFT, 'M': _Keys.RIGHT}.get(ch2, _Keys.OTHER)
        if ch in ('\r', '\n'):
            return _Keys.ENTER
        if ch in ('\x1b',):
            return _Keys.ESC
        if ch.lower() in ('q',):
            return _Keys.QUIT
        return _Keys.OTHER

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == '\x1b':
            seq = sys.stdin.read(2)
            return {'[A': _Keys.UP, '[B': _Keys.DOWN, '[C': _Keys.RIGHT, '[D': _Keys.LEFT}.get(seq, _Keys.ESC)
        if ch in ('\r', '\n'):
            return _Keys.ENTER
        if ch.lower() == 'q':
            return _Keys.QUIT
        return _Keys.OTHER
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _menu(title: str, items: List[Tuple[str, str]]) -> str:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        _die('No TTY detected. Run with subcommands (e.g. `python hoopsctl.py --help`).', code=2)
    _ensure_ansi_on_windows()
    idx = 0
    while True:
        sys.stdout.write('\x1b[2J\x1b[H')
        sys.stdout.write(f'{title}\n\n')
        for i, (_key, label) in enumerate(items):
            prefix = ' > ' if i == idx else '   '
            sys.stdout.write(prefix + label + '\n')
        sys.stdout.write('\n')
        sys.stdout.write('Up/Down to move, Enter to select, q to quit.\n')
        sys.stdout.flush()

        k = _read_key()
        if k == _Keys.UP:
            idx = (idx - 1) % len(items)
        elif k == _Keys.DOWN:
            idx = (idx + 1) % len(items)
        elif k == _Keys.ENTER:
            return items[idx][0]
        elif k in (_Keys.QUIT, _Keys.ESC):
            return 'quit'


def _interactive_main() -> int:
    items = [('start', 'Start training (new run)'), ('resume', 'Resume latest run (in-place)'), ('status', 'Status (active runs / tmux)'), ('list', 'List runs'), ('logs', 'Tail latest run log'), ('eval', 'Eval latest checkpoint of latest run (writes video)'), ('purge', 'Delete ALL runs/videos (asks for confirmation)'), ('doctor', 'Doctor (env sanity checks)'), ('quit', 'Quit')]
    choice = _menu('rl-drone-hoops ops', items)
    if choice == 'quit':
        return 0

    if choice == 'doctor':
        print(json.dumps(doctor(), indent=2))
        return 0
    if choice == 'status':
        print(json.dumps(status(), indent=2))
        return 0
    if choice == 'list':
        print(json.dumps(list_runs(), indent=2))
        return 0

    run_dirs = list_run_dirs(include_discarded=False)
    latest = run_dirs[0].name if run_dirs else ''

    if choice == 'start':
        name = _default_run_name()
        out = start_training(StartOpts(run_name=name, run_dir=RUNS_DIR / name, python=_python_default(), total_steps=200_000, num_envs=4, extra_args=[], resume=False, checkpoint='', attach=False, json_out=False))
        print(json.dumps(out, indent=2))
        return 0

    if choice == 'resume':
        if not latest:
            _die('No existing runs under ./runs to resume.', code=1)
        out = start_training(StartOpts(run_name=latest, run_dir=RUNS_DIR / latest, python=_python_default(), total_steps=200_000, num_envs=4, extra_args=[], resume=True, checkpoint='', attach=False, json_out=False))
        print(json.dumps(out, indent=2))
        return 0

    if choice == 'logs':
        if not latest:
            _die('No runs found.', code=1)
        out = show_logs(run_name=latest, n=60)
        print(out['tail'], end='')
        return 0

    if choice == 'eval':
        if not latest:
            _die('No runs found.', code=1)
        out = eval_latest(run_name=latest, episodes=3, video=True)
        if out.get('stdout'):
            print(out['stdout'], end='')
        if out.get('stderr'):
            print(out['stderr'], file=sys.stderr, end='')
        return 0 if out.get('ok') else 1

    if choice == 'purge':
        print('This will delete ALL of ./runs and ./videos.')
        ans = input('Type DELETE to confirm: ').strip()
        if ans != 'DELETE':
            print('Cancelled.')
            return 0
        out = purge_results(purge_runs=True, purge_videos=True, yes=True)
        print(json.dumps(out, indent=2))
        return 0 if out.get('ok') else 1

    return 0


def _print_json(obj: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(obj, sort_keys=True) + '\n')


def _parse_args(argv: Sequence[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog='hoopsctl.py')
    ap.add_argument('--json', action='store_true', help='Emit machine-readable JSON (where applicable).')
    sub = ap.add_subparsers(dest='cmd')

    sp = sub.add_parser('start', help='Start a new training run (tmux on Linux; WSL+tmux if available on Windows).')
    sp.add_argument('--name', type=str, default='', help='Run name (defaults to ppo_rnn_YYYYMMDD_HHMMSS).')
    sp.add_argument('--run-dir', type=str, default='', help='Explicit run dir (defaults to ./runs/<name>).')
    sp.add_argument('--python', type=str, default='', help='Python interpreter (default: python3 on POSIX, python on Windows).')
    sp.add_argument('--total-steps', type=int, default=200_000)
    sp.add_argument('--num-envs', type=int, default=4)
    sp.add_argument('--resume', action='store_true', help='Resume latest checkpoint in the run dir.')
    sp.add_argument('--checkpoint', type=str, default='', help='Resume from explicit checkpoint path.')
    sp.add_argument('--attach', action='store_true', help='(POSIX tmux only) Attach after starting.')
    sp.add_argument('extra', nargs=argparse.REMAINDER, help='Extra args passed to scripts/train_recurrent_ppo.py (prefix with --).')

    sp = sub.add_parser('stop', help='Stop a run (kill tmux session or PID started by hoopsctl).')
    sp.add_argument('run_name', type=str)

    sub.add_parser('status', help='Show tmux sessions and managed runs.')
    sub.add_parser('list', help='List run directories under ./runs.')

    sp = sub.add_parser('logs', help="Tail a run's train.log.")
    sp.add_argument('run_name', type=str)
    sp.add_argument('-n', type=int, default=60)

    sp = sub.add_parser('eval', help='Eval latest checkpoint for a run (writes MP4 under ./videos by default).')
    sp.add_argument('run_name', type=str)
    sp.add_argument('--episodes', type=int, default=3)
    sp.add_argument('--no-video', action='store_true')

    sp = sub.add_parser('purge', help='Delete runs/videos (DANGEROUS). Requires --yes.')
    sp.add_argument('--runs', action='store_true', help='Delete ./runs/*')
    sp.add_argument('--videos', action='store_true', help='Delete ./videos/*')
    sp.add_argument('--all', action='store_true', help='Delete both runs and videos.')
    sp.add_argument('--yes', action='store_true', help='Actually delete (required).')

    sub.add_parser('doctor', help='Sanity checks (python, imports, tmux, etc.).')

    return ap.parse_args(list(argv))


def main(argv: Optional[Sequence[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        return _interactive_main()

    # Allow --json anywhere (before/after subcommands) for agent ergonomics.
    json_out = "--json" in argv
    if json_out:
        argv = [a for a in argv if a != "--json"]

    ns = _parse_args(argv)
    cmd = getattr(ns, 'cmd', None)
    if cmd is None:
        _die('Missing command. Use --help.', code=2)

    if cmd == 'start':
        name = ns.name or _default_run_name()
        run_dir = Path(ns.run_dir).expanduser() if ns.run_dir else (RUNS_DIR / name)
        py = ns.python or _python_default()
        extra = list(ns.extra or [])
        if extra and extra[0] == '--':
            extra = extra[1:]
        out = start_training(StartOpts(run_name=name, run_dir=run_dir, python=py, total_steps=int(ns.total_steps), num_envs=int(ns.num_envs), extra_args=extra, resume=bool(ns.resume), checkpoint=str(ns.checkpoint or ''), attach=bool(ns.attach), json_out=json_out))
        if json_out:
            _print_json(out)
        else:
            print(json.dumps(out, indent=2))
        return 0 if out.get('ok') else 1

    if cmd == 'stop':
        out = stop_run(run_name=str(ns.run_name), json_out=json_out)
        if json_out:
            _print_json(out)
        else:
            print(json.dumps(out, indent=2))
        return 0 if out.get('ok') else 1

    if cmd == 'status':
        out = status(json_out=json_out)
        if json_out:
            _print_json(out)
        else:
            print(json.dumps(out, indent=2))
        return 0

    if cmd == 'list':
        out = list_runs(json_out=json_out)
        if json_out:
            _print_json(out)
        else:
            print(json.dumps(out, indent=2))
        return 0

    if cmd == 'logs':
        out = show_logs(run_name=str(ns.run_name), n=int(ns.n))
        if json_out:
            _print_json(out)
        else:
            sys.stdout.write(out.get('tail', ''))
        return 0

    if cmd == 'eval':
        out = eval_latest(run_name=str(ns.run_name), episodes=int(ns.episodes), video=not bool(ns.no_video), json_out=json_out)
        if json_out:
            _print_json(out)
        else:
            if out.get('stdout'):
                sys.stdout.write(out['stdout'])
            if out.get('stderr'):
                sys.stderr.write(out['stderr'])
        return 0 if out.get('ok') else 1

    if cmd == 'purge':
        purge_runs = bool(ns.all or ns.runs)
        purge_videos = bool(ns.all or ns.videos)
        if not purge_runs and not purge_videos:
            _die('Nothing selected. Use --runs, --videos, or --all.', code=2)
        out = purge_results(purge_runs=purge_runs, purge_videos=purge_videos, yes=bool(ns.yes))
        if json_out:
            _print_json(out)
        else:
            print(json.dumps(out, indent=2))
        return 0 if out.get('ok') else 1

    if cmd == 'doctor':
        out = doctor(json_out=json_out)
        if json_out:
            _print_json(out)
        else:
            print(json.dumps(out, indent=2))
        return 0 if out.get('ok') else 1

    _die(f'Unknown command: {cmd}', code=2)
    return 2


if __name__ == '__main__':
    raise SystemExit(main())
