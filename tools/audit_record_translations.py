"""Audit English research records against losslessly archived original bytes.

Historical manifest hashes are intentionally not rewritten. A changed English
presentation file is checked against its own digest, while the decoded original
is separately checked against the historical digest. Neither check substitutes
one digest for the other or claims English was used in the original experiment.
"""
import argparse
import ast
import base64
import csv
import hashlib
import io
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'docs/record_translations_20261119'
HAN = re.compile('[\u3400-\u9fff]')
SUFFIXES = {'.md', '.txt', '.csv', '.json', '.html', '.py', '.sh', '.yaml', '.yml', '.toml', '.log', '.ipynb', '.rst', '.tsv', '.tex', '.bib'}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def contains_chinese(path):
    try:
        text = path.read_text()
    except (UnicodeDecodeError, OSError):
        return False
    if path.suffix in {'.json', '.ipynb'}:
        try:
            text = json.dumps(json.loads(text), ensure_ascii=False)
        except ValueError:
            pass
    if HAN.search(text):
        return True
    return path.suffix == '.log' and any(HAN.search(chr(int(m.group(1), 16))) for m in re.finditer(r'\\u([0-9a-fA-F]{4})', text))


class HideText(ast.NodeTransformer):
    def visit_Constant(self, node):
        if isinstance(node.value, str):
            return ast.copy_location(ast.Constant('TEXT'), node)
        return node


def numeric_json(value):
    if isinstance(value, str):
        return 'TEXT'
    if isinstance(value, dict):
        return {k: numeric_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [numeric_json(v) for v in value]
    return value


def audit(restore_to=None):
    archive = json.loads((RECORD / 'ORIGINAL_BYTES.json').read_text())['files']
    ledger = json.loads((RECORD / 'TRANSLATION_MAP.json').read_text())
    if set(archive) != set(ledger['files']):
        raise ValueError('Archive/translation coverage differs')
    checked = []; sources = []; numerical = []
    for name, original in archive.items():
        before = base64.b64decode(original['original_bytes_base64'], validate=True)
        entry = ledger['files'][name]; file = ROOT / name
        if digest(before) != original['original_sha256'] or digest(before) != entry['original_sha256']:
            raise ValueError('Original bytes fail historical SHA-256: ' + name)
        current = file.read_bytes()
        if digest(current) != entry['english_sha256'] or contains_chinese(file):
            raise ValueError('English record changed or contains Chinese: ' + name)
        if file.suffix == '.py':
            a, b = ast.parse(before.decode()), ast.parse(current.decode())
            if ast.dump(HideText().visit(a)) != ast.dump(HideText().visit(b)):
                raise ValueError('Non-text computation changed: ' + name)
            sources.append(name)
        if file.suffix == '.json':
            if numeric_json(json.loads(before)) != numeric_json(json.loads(current)):
                raise ValueError('JSON structure/numeric values changed: ' + name)
            numerical.append(name)
        if file.suffix == '.csv':
            a = list(csv.reader(io.StringIO(before.decode())))
            b = list(csv.reader(io.StringIO(current.decode())))
            if len(a) != len(b) or any(len(x) != len(y) for x, y in zip(a, b)):
                raise ValueError('CSV dimensions changed: ' + name)
            for left, right in zip(a, b):
                for x, y in zip(left, right):
                    try:
                        float(x)
                    except ValueError:
                        continue
                    if x != y:
                        raise ValueError('Numeric CSV cell changed: ' + name)
            numerical.append(name)
        checked.append(name)
    remaining = []
    for file in ROOT.rglob('*'):
        if not file.is_file() or any(p in {'.git', '__pycache__', '.pytest_cache', '.mypy_cache'} for p in file.parts):
            continue
        if file.suffix.lower() in SUFFIXES and file.stat().st_size <= 10_000_000 and contains_chinese(file):
            remaining.append(str(file.relative_to(ROOT)))
    if remaining:
        raise ValueError('Untranslated project text: ' + repr(remaining))
    # Identify changed presentation entries while preserving every original manifest.
    historical_entries = []
    for file in ROOT.rglob('*MANIFEST.json'):
        if not file.is_file() or '.git' in file.parts:
            continue
        try:
            manifest = json.loads(file.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(manifest, dict):
            continue
        for rel, expected in manifest.items():
            if not isinstance(expected, str) or not re.fullmatch('[a-f0-9]{64}', expected):
                continue
            path = (file.parent / rel).resolve()
            if not path.is_relative_to(ROOT):
                continue
            name = str(path.relative_to(ROOT))
            if name in archive:
                if expected != archive[name]['original_sha256']:
                    raise ValueError('Historical translated-file hash has an unexpected origin: ' + name)
                historical_entries.append({'manifest': str(file.relative_to(ROOT)), 'path': name, 'historical_sha256_preserved': True})
    protected = json.loads((RECORD / 'SCIENTIFIC_INPUTS.json').read_text())
    for name, expected in protected.items():
        if digest(Path(name).read_bytes()) != expected:
            raise ValueError('Scientific input changed: ' + name)
    if restore_to is not None:
        restore_to = restore_to.resolve()
        if restore_to.is_relative_to(ROOT):
            raise ValueError('Restore historical-language bytes outside the active project')
        if restore_to.exists():
            raise FileExistsError('Restore only into a new directory')
        restore_to.mkdir(parents=True)
        for name, record in archive.items():
            destination = restore_to / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(record['original_bytes_base64']))
    result = {'status': 'passed', 'translated_files': len(checked),
              'original_archive_sha256': digest((RECORD / 'ORIGINAL_BYTES.json').read_bytes()),
              'translation_map_sha256': digest((RECORD / 'TRANSLATION_MAP.json').read_bytes()),
              'audit_source_sha256': digest(Path(__file__).read_bytes()),
              'non_text_source_ASTs_unchanged': sources,
              'numeric_JSON_and_CSV_records_unchanged': numerical, 'remaining_Chinese_text_files': remaining,
              'historical_manifest_entries_verified_against_archived_originals': historical_entries,
              'scientific_inputs_unchanged': len(protected), 'original_bytes_restorable': True,
              'restored_to': str(restore_to) if restore_to else None,
              'scope': 'English presentation and original-byte provenance; does not rerun training, inference, IK or old numerical experiments'}
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--restore-originals-to', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = audit(args.restore_originals_to)
    if args.output:
        if args.output.exists():
            raise FileExistsError('Do not overwrite an existing audit result')
        args.output.write_text(json.dumps(result, indent=2) + '\n')
