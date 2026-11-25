# Research records use English

Effective 2026-11-19, at Alex's request, project documentation, research-log entries, tables, generated reports and HTML overviews use English for recording and group sharing. Existing experiment dates, choices, negative findings and numerical results remain historical facts. Translation is a presentation change, not a new experiment or a new execution of an old one.

The migration includes readable records under `docs/`, local `outputs/`, generated-report strings, matching historical text/source snapshots, and metric labels stored as escaped Unicode in JSON/logs. Raw data, checkpoints, prediction arrays, optimization settings, metric implementations and animations are unchanged.

## Original evidence and hashes

[ORIGINAL_BYTES.json](record_translations_20261119/ORIGINAL_BYTES.json) preserves every translated file's pre-translation bytes losslessly as base64, with its SHA-256. [TRANSLATION_MAP.json](record_translations_20261119/TRANSLATION_MAP.json) identifies the current English counterpart and checksum. Original-language bytes are an encoded archival record, not the active reading copy.

Historical experiment protocols and manifests retain their original hashes. English presentation files and translated source snapshots have different bytes; they must not be claimed to have run in English during the original experiment. A legacy validator checking current files against historical hashes may correctly report a presentation/source-snapshot change. The translation audit separately verifies both the archived originals and current English versions, unchanged numerical JSON/CSV values, and identical non-string computation ASTs for localized generators.

```bash
python tools/audit_record_translations.py
```

To inspect the exact historical text/source bytes, restore them into a **new** directory; the active project is not overwritten:

```bash
python tools/audit_record_translations.py \
  --restore-originals-to /tmp/imuposer-records-before-english
```

This restores the translated files, not a complete dataset or executable environment. Historical protocols contain their original absolute paths. Full experiment reproduction still requires the licensed assets, caches and execution environment described in the original reports. Original manifests are not silently rewritten to make a different presentation look like the original execution.

[Migration audit](record_translations_20261119/VALIDATION.json) records the final checks. Future documentation/report generators should keep their visible prose in English. Keep temporal IK as a separate extension from the paper-guided framewise baseline.
