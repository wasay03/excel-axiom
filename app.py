import io
from typing import Optional

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Excel Axiom", layout="wide")


def read_table_from_upload(uploaded) -> Optional[pd.DataFrame]:

	if uploaded is None:
		return None

	name = uploaded.name.lower()
	data = uploaded.read()
	buffer = io.BytesIO(data)

	if name.endswith(".csv"):
		buffer.seek(0)
		return pd.read_csv(buffer)
	elif name.endswith(".xlsx") or name.endswith(".xls"):
		buffer.seek(0)
		# engine auto-detect; ensure openpyxl/xlrd installed
		return pd.read_excel(buffer)
	else:
		st.error("Unsupported file type. Please upload .xlsx, .xls, or .csv")
		return None


def _split_list_like(value, case_insensitive: bool) -> set:

	if pd.isna(value):
		return set()

	if isinstance(value, (list, tuple, set)):
		items = list(value)
	else:
		text = str(value)
		# Common separators in annotations
		for sep in [";", ",", "|", "/"]:
			text = text.replace(sep, "|")
		items = [part.strip() for part in text.split("|") if part.strip() != ""]

	if case_insensitive:
		items = [str(x).lower() for x in items]

	return set(items)


def _values_equal(a, b, *,
	case_insensitive: bool,
	trim_whitespace: bool,
	numeric_tolerance: float,
	list_as_set: bool,
	na_empty_equal: bool,
) -> bool:

	# NA handling
	if na_empty_equal:
		if (a is None or (isinstance(a, float) and pd.isna(a)) or (isinstance(a, str) and a.strip() == "")) and \
			(b is None or (isinstance(b, float) and pd.isna(b)) or (isinstance(b, str) and b.strip() == "")):
			return True

	# Both NaN
	if pd.isna(a) and pd.isna(b):
		return True

	# Numeric tolerance
	try:
		fa = float(a)
		fb = float(b)
		return abs(fa - fb) <= numeric_tolerance
	except Exception:
		pass

	# List-like comparison
	if list_as_set:
		set_a = _split_list_like(a, case_insensitive)
		set_b = _split_list_like(b, case_insensitive)
		return set_a == set_b

	# Text normalization
	if isinstance(a, str) or isinstance(b, str):
		sa = "" if a is None else str(a)
		sb = "" if b is None else str(b)
		if trim_whitespace:
			sa = sa.strip()
			sb = sb.strip()
		if case_insensitive:
			sa = sa.lower()
			sb = sb.lower()
		return sa == sb

	# Fallback direct equality
	return a == b



def main() -> None:

	st.title("Excel Axiom – Interactive Excel Filter & Compare Tool")

	with st.sidebar:
		st.header("Upload")
		uploaded = st.file_uploader(
			"Upload Excel/CSV file",
			type=["xlsx", "xls", "csv"],
			accept_multiple_files=False,
		)

		st.markdown("---")
		st.caption("The first row is treated as column headers.")

	df = read_table_from_upload(uploaded)

	if df is None:
		st.info("Upload an Excel/CSV file from the sidebar to begin.")
		return

	# Initialize session state
	if "filters" not in st.session_state:
		st.session_state["filters"] = {}
	if "base_df" not in st.session_state:
		st.session_state["base_df"] = df.copy()
		st.session_state["working_df"] = df.copy()
		st.session_state["last_upload_name"] = uploaded.name if uploaded else None
	else:
		# Reset state if a new file is uploaded
		current_name = uploaded.name if uploaded else None
		if st.session_state.get("last_upload_name") != current_name:
			st.session_state["base_df"] = df.copy()
			st.session_state["working_df"] = df.copy()
			st.session_state["filters"] = {}
			st.session_state["last_upload_name"] = current_name

	working_df = st.session_state["working_df"]

	st.success(f"Loaded primary file with {len(working_df):,} rows and {len(working_df.columns):,} columns.")

	# New Column Section (placed before filters so new columns are reflected immediately)
	st.markdown("---")
	st.subheader("Add New Column")
	with st.expander("Create column", expanded=False):
		mode = st.radio("Mode", options=["Constant value", "Expression"], horizontal=True)
		new_col_name = st.text_input("New column name")
		if mode == "Constant value":
			val_type = st.selectbox("Value type", options=["Text", "Number", "Boolean"])
			if val_type == "Text":
				const_val = st.text_input("Value", value="")
			elif val_type == "Number":
				const_val = st.number_input("Value", value=0.0)
			else:
				const_val = st.checkbox("Value", value=False)
			if st.button("Add column", type="primary", disabled=(new_col_name.strip() == "")):
				wd = st.session_state["working_df"].copy()
				wd[new_col_name] = const_val
				st.session_state["working_df"] = wd
				st.success(f"Added column '{new_col_name}'.")

		else:
			expr = st.text_input("Expression (pandas eval)", placeholder="e.g. Quantity * Price")
			st.caption("Uses pandas.DataFrame.eval. You can reference existing column names directly.")
			if st.button("Add column from expression", type="primary", disabled=(new_col_name.strip() == "" or expr.strip() == "")):
				try:
					wd = st.session_state["working_df"].copy()
					wd[new_col_name] = wd.eval(expr)
					st.session_state["working_df"] = wd
					st.success(f"Added column '{new_col_name}' from expression.")

				except Exception as e:
					st.error(f"Failed to evaluate expression: {e}")

	with st.sidebar:
		st.header("Filters")
		with st.expander("Per-Column Filters", expanded=True):
			# Keep UI widgets and saved filters in sync (no callbacks)
			select_all_clicked = st.button("Select all", type="secondary")
			clear_all_clicked = st.button("Clear all filters", type="secondary")
			if select_all_clicked or clear_all_clicked:
				new_filters = {}
				for c in working_df.columns:
					if select_all_clicked:
						vals = working_df[c].dropna().unique().tolist()
						try:
							vals = sorted(vals)
						except Exception:
							pass
						new_filters[c] = vals
					else:
						new_filters[c] = []
				st.session_state["filters"] = new_filters

			for column_name in working_df.columns:
				series = working_df[column_name]
				# Convert to string for display consistency while keeping original values for filtering
				unique_values = series.dropna().unique().tolist()
				try:
					unique_values_sorted = sorted(unique_values)
				except Exception:
					unique_values_sorted = unique_values

				default_selected = st.session_state["filters"].get(column_name, unique_values_sorted)

				selected = st.multiselect(
					f"{column_name}", options=unique_values_sorted, default=default_selected
				)
				st.session_state["filters"][column_name] = selected

	# Edit data grid (add rows / edit values)
	st.markdown("---")
	st.subheader("Edit Data (optional)")
	edited_df = st.data_editor(
		working_df,
		use_container_width=True,
		num_rows="dynamic",
	)
	if st.button("Apply edits"):
		# Save edits and normalize filters to available values
		st.session_state["working_df"] = edited_df.copy()
		normalized = {}
		for c in edited_df.columns:
			available = edited_df[c].dropna().unique().tolist()
			try:
				available = sorted(available)
			except Exception:
				pass
			prev = st.session_state["filters"].get(c, available)
			normalized[c] = [v for v in prev if v in available] or available
		st.session_state["filters"] = normalized

	# Apply filters
	filtered_df = working_df.copy()
	for column_name, selected_values in st.session_state["filters"].items():
		if selected_values is None:
			continue
		if len(selected_values) == 0:
			# If nothing selected for this column, skip it (don't filter)
			continue
		filtered_df = filtered_df[filtered_df[column_name].isin(selected_values)]

	st.subheader("Filtered Data")
	st.dataframe(filtered_df, use_container_width=True)
	st.caption(f"Rows: {len(filtered_df):,} | Columns: {len(filtered_df.columns):,}")

	# Downloads
	def to_excel_bytes(dataframe: pd.DataFrame) -> bytes:
		output = io.BytesIO()
		with pd.ExcelWriter(output, engine="openpyxl") as writer:
			dataframe.to_excel(writer, index=False, sheet_name="Filtered")
		return output.getvalue()

	col_csv, col_xlsx = st.columns(2)
	with col_csv:
		st.download_button(
			label="Download CSV",
			data=filtered_df.to_csv(index=False).encode("utf-8"),
			file_name="filtered.csv",
			mime="text/csv",
		)
	with col_xlsx:
		st.download_button(
			label="Download Excel",
			data=to_excel_bytes(filtered_df),
			file_name="filtered.xlsx",
			mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
		)


if __name__ == "__main__":
	main()


