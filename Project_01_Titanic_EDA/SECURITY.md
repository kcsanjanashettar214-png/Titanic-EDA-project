# Security and Data Protection

This project is a local, educational data-analysis project. It does not provide a network service, accept browser uploads, authenticate users, or encrypt files. The included Titanic dataset is public historical data; the safeguards below reduce accidental misuse and unsafe file handling but are not a security boundary against someone who can access the project folder or computer.

## Input validation

- The analysis accepts `.csv` files only.
- Input files must resolve to a regular file inside the project's `data/` directory. Absolute paths, `..` traversal, and symlinks that resolve outside that directory are rejected.
- CSV files are limited to 10 MB to bound memory use.
- Input must be readable UTF-8 CSV with a header row and unique column names.
- Required analysis columns are checked before analysis: `survived`, `pclass`, `sex`, `age`, `fare`, and `embarked`. Required numeric values and supported survival, class, and gender values are validated.
- Empty, malformed, unreadable, unsupported, or invalid files stop analysis with a short error that does not include the full path, file contents, or exception details.

## File and data handling

- The output directory must remain inside the project's `outputs/` directory; path traversal and paths resolving outside it are rejected.
- Loading data reads the source CSV; `clean_dataset()` works on a copy. The original `data/titanic.csv` is not overwritten.
- The cleaned CSV and generated charts are analysis outputs. Avoid including them when sharing only the aggregate dashboard.
- The browser dashboard shows aggregate findings and charts and does not link to row-level CSV data. This does not prevent someone with access to the project folder from opening the source or output files.
- No passwords, API keys, or credentials are required or stored by the project.

## Dependencies and maintenance

- Runtime and notebook dependencies are the established Pandas, NumPy, Matplotlib, Seaborn, and Jupyter packages listed in `requirements.txt`, installed through pip from their configured package index.
- Use a virtual environment and periodically update dependencies with `python -m pip install --upgrade -r requirements.txt`. Review dependency changes before using the project in a sensitive environment.
- Do not install packages or run analysis on untrusted CSV files in a privileged environment.

## Limitations

These controls provide basic input validation and path containment for normal local use. They do not encrypt data, enforce operating-system permissions, defend against a malicious process modifying files concurrently, or protect data shared in the project folder. Use operating-system access controls and encryption if genuinely sensitive data is introduced.
