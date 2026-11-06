#!/usr/bin/env python3
"""Interleaved A/B with raw evidence, exact stdout checks and an optional decode gate."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import resource
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
BATCH_PROFILE_COUNTERS = ('batch_decode', 'batch_qkv', 'batch_attention', 'batch_ffn', 'batch_lm_head')


def parse(err):
    d = {}
    for key, pattern in (
        ('infer', r'Inference: ([\d.]+) ms'),
        ('prefill', r'Prefill: \d+ tokens \(([\d.]+) ms\)'),
        ('encoder', r'  Encoder: \d+ tokens \(([\d.]+) ms\)'),
    ):
        values = re.findall(pattern, err)
        if values:
            d[key] = sum(map(float, values))
    dec = re.findall(r'Decode: (\d+) tokens \(([\d.]+) ms, ([\d.]+) ms/token\)', err)
    if dec:
        d['dec_ms'] = sum(float(x[1]) for x in dec)
        d['dec_tok'] = sum(int(x[0]) for x in dec)
        if d['dec_tok']:
            d['ms_tok'] = d['dec_ms'] / d['dec_tok']
    for name in BATCH_PROFILE_COUNTERS:
        values = re.findall(rf'\[profile\] {name}: ([\d.]+)ms \((\d+) calls,', err)
        if values:
            d[f'{name}_ms'] = sum(float(ms) for ms, _ in values)
            d[f'{name}_calls'] = sum(int(calls) for _, calls in values)
    return d


def cases(fixtures, aligner):
    sample = ROOT / 'bench/samples/audio.wav'
    long = ROOT / 'bench/long/samples'
    return {
        'wav_offline': (['-i', str(sample)], None),
        'c28_offline': (['-i', str(fixtures / 'clip28s.wav')], None),
        'c4_offline': (['-i', str(fixtures / 'clip4s.wav')], None),
        'wav_seg30': (['-i', str(sample), '-S', '30'], None),
        'wav_stream': (['-i', str(sample), '--stream'], None),
        'mixed_offline': (['-i', str(ROOT / 'bench/samples/mixed_zh_en.wav')], None),
        '2min_offline': (['-i', str(long / 'long-2min.wav')], None),
        '10min_offline': (['-i', str(long / 'long-10min.wav')], None),
        '2min_seg30': (['-i', str(long / 'long-2min.wav'), '-S', '30'], None),
        '10min_seg30': (['-i', str(long / 'long-10min.wav'), '-S', '30'], None),
        '2min_stream': (['-i', str(long / 'long-2min.wav'), '--stream'], None),
        '10min_stream': (['-i', str(long / 'long-10min.wav'), '--stream'], None),
        'push_c28': (['--stdin', '--stream'], fixtures / 'clip28s.wav'),
        'push_2min': (['--stdin', '--stream'], long / 'long-2min.wav'),
        'multi12': (['-i', ','.join(str(fixtures / f'multi/m{i:02d}.wav') for i in range(12))], None),
        'multi60': (['-i', ','.join(str(fixtures / f'multi60/m{i:02d}.wav') for i in range(60))], None),
        'srt_2min': (['-i', str(long / 'long-2min.wav'), '-S', '30', '--aligner-dir', str(aligner)], None),
    }


def snapshot():
    result = {'time': time.time(), 'loadavg': os.getloadavg()}
    commands = {'processes': ['ps', '-A', '-o', 'pid,pcpu,comm', '-r']}
    if platform.system() == 'Darwin':
        commands.update(power=['pmset', '-g', 'batt'], thermal=['pmset', '-g', 'therm'],
                        vm_stat=['vm_stat'], swap_usage=['sysctl', 'vm.swapusage'])
    for name, cmd in commands.items():
        p = subprocess.run(cmd, capture_output=True, text=True, check=False)
        result[name] = '\n'.join(p.stdout.splitlines()[:16]) if name == 'processes' else p.stdout.strip()
    return result


def run(binary, args, stdin_path, env, prefix, timeout):
    cmd = [str(binary), *args]
    before = snapshot()
    start = time.monotonic()
    timed_out = False
    with (open(stdin_path, 'rb') if stdin_path else open(os.devnull, 'rb')) as fin:
        usage_before = resource.getrusage(resource.RUSAGE_CHILDREN)
        try:
            p = subprocess.run(cmd, stdin=fin, capture_output=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            p = subprocess.CompletedProcess(cmd, 124, exc.stdout or b'', exc.stderr or b'')
        usage_after = resource.getrusage(resource.RUSAGE_CHILDREN)
    wall = (time.monotonic() - start) * 1000
    prefix.with_suffix('.stdout').write_bytes(p.stdout)
    prefix.with_suffix('.stderr').write_bytes(p.stderr)
    row = {'cmd': cmd, 'before': before, 'after': snapshot(), 'returncode': p.returncode, 'timed_out': timed_out,
           'stdout_sha256': hashlib.sha256(p.stdout).hexdigest(),
           'resource_usage': {key: getattr(usage_after, key) - getattr(usage_before, key)
                              for key in ('ru_utime', 'ru_stime', 'ru_minflt', 'ru_majflt', 'ru_nvcsw', 'ru_nivcsw')},
           'metrics': {**parse(p.stderr.decode('utf-8', 'replace')), 'wall': wall}}
    prefix.with_suffix('.json').write_text(json.dumps(row, indent=2) + '\n')
    if p.returncode:
        raise RuntimeError(f'{prefix}: exit {p.returncode}; see .stderr')
    if not p.stdout.strip() and '--json' not in args:
        raise RuntimeError(f'{prefix}: empty transcript')
    return row


def summarize(rows, max_decode_regression=None):
    summaries = {}
    failed = False
    for case in dict.fromkeys(r['case'] for r in rows):
        all_rows = [r for r in rows if r['case'] == case]
        identical = len({r['stdout_sha256'] for r in all_rows}) == 1
        # JSON alignment output, when requested, must also remain byte-identical.
        structured = {r['structured_sha256'] for r in all_rows if 'structured_sha256' in r}
        identical = identical and len(structured) <= 1
        failed |= not identical
        measured = [r for r in all_rows if not r['warmup']]
        result = {'identical': identical, 'metrics': {}}
        print(f'== {case}: exact outputs={identical}', flush=True)
        keys = ('wall', 'infer', 'encoder', 'prefill', 'dec_ms', 'dec_tok', 'ms_tok')
        keys += tuple(f'{name}_{unit}' for name in BATCH_PROFILE_COUNTERS for unit in ('ms', 'calls'))
        for key in keys:
            if not all(key in r['metrics'] for r in measured):
                continue
            a, b = ([r['metrics'][key] for r in measured if r['who'] == who] for who in ('A', 'B'))
            am, bm = statistics.median(a), statistics.median(b)
            delta = (bm / am - 1) * 100 if am else 0.0
            paired = [(bv / av - 1) * 100 for av, bv in zip(a, b) if av != 0]
            paired_median = statistics.median(paired) if paired else None
            result['metrics'][key] = {'A': a, 'B': b, 'A_median': am, 'B_median': bm, 'delta_pct': delta,
                                      'paired_delta_pct': paired, 'paired_median_delta_pct': paired_median}
            paired_text = f'{paired_median:+.2f}%' if paired_median is not None else 'n/a'
            print(f'  {key:8} A={am:.3f} B={bm:.3f} ({delta:+.2f}%; paired median {paired_text})', flush=True)
        if max_decode_regression is not None:
            metric = result['metrics'].get('ms_tok')
            result['decode_gate_passed'] = metric is not None and metric['delta_pct'] <= max_decode_regression
            failed |= not result['decode_gate_passed']
        summaries[case] = result
    return summaries, failed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('a', type=Path)
    ap.add_argument('b', type=Path)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--threads', type=int, default=12)
    ap.add_argument('--profile', action='store_true', help='Collect batch phase diagnostics; adds profiling overhead')
    ap.add_argument('--env', action='append', default=[])
    ap.add_argument('--a-env', action='append', default=[])
    ap.add_argument('--b-env', action='append', default=[])
    ap.add_argument('--model', type=Path, default=ROOT / 'qwen3-asr-0.6b')
    ap.add_argument('--aligner', type=Path, default=ROOT / 'qwen3-aligner-0.6b')
    ap.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parent / 'fixtures')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--timeout', type=float, default=600)
    ap.add_argument('--max-decode-regression', type=float, help='Fail if median B ms/token exceeds A by this percent')
    o = ap.parse_args()
    if o.runs < 1 or o.threads < 1 or o.timeout <= 0:
        ap.error('runs, threads and timeout must be positive')
    definitions = cases(o.fixtures, o.aligner)
    selected = o.cases.split(',')
    if any(c not in definitions for c in selected) or len(set(selected)) != len(selected):
        ap.error(f'cases must be distinct members of {list(definitions)}')
    envs = {}
    # Avoid inherited tuning switches silently changing one comparison.
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith('QWEN_ASR_')}
    for who, additions in (('A', o.a_env), ('B', o.b_env)):
        try:
            envs[who] = {**clean_env, **dict(x.split('=', 1) for x in o.env + additions)}
        except ValueError:
            ap.error('environment overrides must be KEY=VALUE')
    binaries = {'A': o.a.resolve(), 'B': o.b.resolve()}
    hashes = {who: hashlib.sha256(p.read_bytes()).hexdigest() for who, p in binaries.items()}
    o.out.mkdir(parents=True, exist_ok=False)
    metadata = {'platform': platform.platform(), 'model': str(o.model.resolve()),
                'cases': selected, 'fixtures': str(o.fixtures.resolve()), 'aligner': str(o.aligner.resolve()), 'binaries': {k: str(v) for k, v in binaries.items()},
                'binary_sha256': hashes, 'threads': o.threads, 'runs': o.runs, 'profile': o.profile,
                'overrides': {'common': o.env, 'A': o.a_env, 'B': o.b_env},
                'git_head': subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()}
    (o.out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    rows = []
    for case in selected:
        args, stdin_path = definitions[case]
        for rep in range(o.runs + 1):
            order = ('A', 'B') if rep == 0 or rep % 2 == 1 else ('B', 'A')
            for who in order:
                prefix = o.out / f'{case}-{rep}-{who}'
                cmd = ['-d', str(o.model), *args, '-t', str(o.threads), '--debug']
                if o.profile:
                    cmd.append('--profile')
                structured = prefix.with_suffix('.alignment.json')
                if case == 'srt_2min':
                    cmd += ['--json', str(structured)]
                row = run(binaries[who], cmd, stdin_path, envs[who], prefix, o.timeout)
                row.update(case=case, who=who, warmup=rep == 0)
                if case == 'srt_2min':
                    data = structured.read_bytes()
                    json.loads(data)
                    row['structured_sha256'] = hashlib.sha256(data).hexdigest()
                rows.append(row)
                with (o.out / 'runs.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(f'{case} {"warmup" if rep == 0 else rep} {who}: {row["metrics"]}', flush=True)
    summary, failed = summarize(rows, o.max_decode_regression)
    (o.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    raise SystemExit(1 if failed else 0)


if __name__ == '__main__':
    main()
