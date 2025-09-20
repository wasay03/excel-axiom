# Excel Axiom – Interactive Excel Filter & Compare Tool

## Features

- Upload an Excel file (`.xlsx`, `.xls`, `.csv`) and preview data
- Automatic per-column filters based on unique values
- Persistent filter selections while you explore
- Add new columns (manual or formula/expression based)
- Upload a control file and compare by key columns (added/removed/changed)
- Export filtered and comparison results to Excel/CSV

## Quick Start

1. Install Python 3.9+
2. Create a virtual environment (recommended)
3. Install dependencies
4. Run the app

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Usage

1. Upload your primary Excel/CSV file in the sidebar.
2. Use the dynamic filters (per column) to refine the dataset.
3. Optionally, create new columns:
   - Manual: fill a constant or derive from an existing column with a simple operation.
   - Expression: use pandas-style expressions, e.g. `Total = Quantity * Price`.
4. Upload a control file to compare by one or more key columns:
   - See rows Added (in primary only), Removed (in control only), and Changed (keys match, non-key values differ).
5. Download filtered and comparison outputs from the Download section.

## Notes

- The first row of your Excel/CSV is treated as the header.
- Very large files may take time to load; filters are computed from unique values per column.
- Expressions use `pandas.DataFrame.eval`. Avoid unsafe code; only DataFrame operations are allowed.


