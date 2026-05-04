# Public Data Mapping Summary

The project was tested with downloaded public sample data mapped into the
training schemas by `src.prepare_public_data`.

## Local Mapped Outputs

| File | Rows | GitHub status |
|------|------|---------------|
| `data_public/claims.csv` | 66,494 | gitignored generated artifact |
| `data_public/labs.csv` | 5,928 | gitignored generated artifact |
| `data_public/notes.csv` | 222 | gitignored generated artifact |

The public CSVs are intentionally not committed. This summary records the
verified output counts while keeping the repository lightweight.

## Command

```bash
python -m src.prepare_public_data --raw-dir tmp/raw_public --out data_public
```
