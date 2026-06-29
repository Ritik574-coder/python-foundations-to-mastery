# 🐼 Pandas Comprehensive Cheat Sheet

> **Pandas version:** 2.x | **Python:** 3.9+

---

## Table of Contents
1. [Installation & Import](#1-installation--import)
2. [Core Data Structures](#2-core-data-structures)
3. [Creating DataFrames](#3-creating-dataframes)
4. [Reading & Writing Data (I/O)](#4-reading--writing-data-io)
5. [Inspection & Info](#5-inspection--info)
6. [Selection & Indexing](#6-selection--indexing)
7. [Filtering & Querying](#7-filtering--querying)
8. [Data Cleaning](#8-data-cleaning)
9. [Data Types & Casting](#9-data-types--casting)
10. [String Operations (str accessor)](#10-string-operations-str-accessor)
11. [DateTime Operations (dt accessor)](#11-datetime-operations-dt-accessor)
12. [Sorting & Ranking](#12-sorting--ranking)
13. [GroupBy & Aggregation](#13-groupby--aggregation)
14. [Merging, Joining & Concatenating](#14-merging-joining--concatenating)
15. [Reshaping Data](#15-reshaping-data)
16. [Apply, Map & Transform](#16-apply-map--transform)
17. [Window Functions](#17-window-functions)
18. [MultiIndex](#18-multiindex)
19. [Categorical Data](#19-categorical-data)
20. [Plotting](#20-plotting)
21. [Performance & Memory Optimization](#21-performance--memory-optimization)
22. [Pandas Options & Settings](#22-pandas-options--settings)
23. [Common Patterns & Recipes](#23-common-patterns--recipes)

---

## 1. Installation & Import

```python
pip install pandas                  # latest stable
pip install pandas==2.2.2           # specific version
pip install pandas[performance]     # with optional speed deps (numexpr, bottleneck)
pip install pandas pyarrow          # for Parquet / Arrow backend

import pandas as pd
import numpy as np                  # almost always used together

pd.__version__                      # check version
```

---

## 2. Core Data Structures

### Series
A 1-D labeled array — think of it as a single column.

```python
s = pd.Series([10, 20, 30], index=['a', 'b', 'c'], name='scores')

s.values          # numpy array: array([10, 20, 30])
s.index           # Index(['a', 'b', 'c'])
s.dtype           # dtype('int64')
s.name            # 'scores'
s.shape           # (3,)
s.size            # 3
s['b']            # 20  — label-based access
s[1]              # 20  — position-based access
```

### DataFrame
A 2-D labeled table — the workhorse of pandas.

```python
df = pd.DataFrame({
    'name':  ['Alice', 'Bob', 'Carol'],
    'age':   [30, 25, 35],
    'score': [88.5, 92.0, 79.3]
})

df.shape          # (3, 3) — (rows, cols)
df.dtypes         # dtype of each column
df.columns        # column labels
df.index          # row labels
df.values         # numpy 2-D array (avoid for mixed types — use df.to_numpy())
df.to_numpy()     # preferred over .values
```

---

## 3. Creating DataFrames

```python
# From dict of lists
pd.DataFrame({'a': [1, 2], 'b': [3, 4]})

# From list of dicts (row-oriented)
pd.DataFrame([{'a': 1, 'b': 3}, {'a': 2, 'b': 4}])

# From list of tuples
pd.DataFrame([(1, 3), (2, 4)], columns=['a', 'b'])

# From NumPy array
pd.DataFrame(np.random.randn(5, 3), columns=['x', 'y', 'z'])

# Empty DataFrame with schema
pd.DataFrame(columns=['id', 'name', 'value'])

# From a single Series
pd.DataFrame(s, columns=['col_name'])

# From a range (useful for scaffolding)
pd.DataFrame({'n': range(1_000_000)})

# Copy an existing DataFrame
df2 = df.copy()               # deep copy (safe)
df2 = df.copy(deep=False)     # shallow copy (shares data)
```

---

## 4. Reading & Writing Data (I/O)

### CSV

```python
# Read
df = pd.read_csv('file.csv')
df = pd.read_csv('file.csv',
    sep=',',                        # delimiter (use sep='\t' for TSV)
    header=0,                       # row number(s) for header
    index_col='id',                 # use column as index
    usecols=['a', 'b', 'c'],        # load only these columns  ← HUGE memory win
    dtype={'id': 'int32', 'val': 'float32'},  # force dtypes at load time
    parse_dates=['date_col'],       # auto-parse date columns
    nrows=10_000,                   # read first N rows only
    skiprows=[1, 2],                # skip specific rows
    na_values=['NA', 'N/A', '--'],  # custom NA strings
    low_memory=False,               # safer dtype inference on big files
    encoding='utf-8',
    chunksize=100_000               # returns TextFileReader iterator
)

# Chunked read for large files
chunks = []
for chunk in pd.read_csv('big.csv', chunksize=100_000):
    chunks.append(chunk[chunk['value'] > 0])
df = pd.concat(chunks, ignore_index=True)

# Write
df.to_csv('out.csv', index=False)
df.to_csv('out.csv', index=False, float_format='%.4f', date_format='%Y-%m-%d')
```

### Excel

```python
# Read
df = pd.read_excel('file.xlsx', sheet_name='Sheet1')
df = pd.read_excel('file.xlsx', sheet_name=0, usecols='A:D', header=1)

# Read all sheets at once
all_sheets = pd.read_excel('file.xlsx', sheet_name=None)  # dict of DataFrames

# Write
df.to_excel('out.xlsx', sheet_name='Results', index=False)

# Write multiple sheets
with pd.ExcelWriter('out.xlsx', engine='openpyxl') as writer:
    df1.to_excel(writer, sheet_name='Summary', index=False)
    df2.to_excel(writer, sheet_name='Detail',  index=False)
```

### JSON

```python
df = pd.read_json('file.json')
df = pd.read_json('file.json', orient='records', lines=True)  # JSON Lines (JSONL)
df.to_json('out.json', orient='records', indent=2)
df.to_json('out.json', orient='records', lines=True)           # JSONL
```

### Parquet  ← Preferred format for data engineering

```python
# pip install pyarrow   OR   pip install fastparquet
df = pd.read_parquet('file.parquet')
df = pd.read_parquet('file.parquet', columns=['a', 'b'])       # column pruning
df = pd.read_parquet('folder/')                                 # partitioned dataset

df.to_parquet('out.parquet', index=False)
df.to_parquet('out.parquet', compression='snappy')             # snappy / gzip / brotli
df.to_parquet('out.parquet', partition_cols=['year', 'month']) # partitioned write
```

### SQL / Database

```python
from sqlalchemy import create_engine

engine = create_engine('postgresql+psycopg2://user:pw@host:5432/db')

# Read
df = pd.read_sql('SELECT * FROM orders WHERE status = %s', engine, params=['open'])
df = pd.read_sql_table('customers', engine, schema='public')
df = pd.read_sql_query('SELECT id, name FROM users LIMIT 1000', engine)

# Write
df.to_sql('table_name', engine, if_exists='append', index=False, chunksize=1000)
# if_exists: 'fail' | 'replace' | 'append'
```

### Other Formats

```python
df = pd.read_feather('file.feather')          # very fast columnar format
df.to_feather('out.feather')

df = pd.read_orc('file.orc')                  # Apache ORC
df = pd.read_html('https://url.com')[0]       # scrape first HTML table on page
df = pd.read_clipboard()                      # paste from clipboard (great for debugging)
df = pd.read_pickle('file.pkl')               # Python pickle (not for long-term storage)
df.to_pickle('out.pkl')
```

---

## 5. Inspection & Info

```python
df.head(10)                 # first 10 rows (default 5)
df.tail(10)                 # last 10 rows
df.sample(5)                # 5 random rows
df.sample(frac=0.01)        # 1% random sample

df.shape                    # (rows, cols)
df.ndim                     # 2
df.size                     # total elements = rows × cols

df.info()                   # dtypes, non-null counts, memory usage
df.info(memory_usage='deep') # exact memory (slower but accurate)
df.memory_usage(deep=True)  # per-column bytes

df.describe()               # stats for numeric cols
df.describe(include='all')  # stats for ALL cols
df.describe(include=[np.number])
df.describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.99])

df.dtypes                   # dtype of each column
df.columns.tolist()         # list of column names
df.index                    # row index
len(df)                     # number of rows

df.nunique()                # count of unique values per column
df.nunique(axis=1)          # per row
df['col'].value_counts()             # frequency table
df['col'].value_counts(normalize=True)  # proportions
df['col'].value_counts(dropna=False)    # include NaN counts

df.isnull().sum()           # missing count per column
df.isnull().sum() / len(df) # missing percentage
df.notnull().all()          # True if no nulls in each col

df.duplicated().sum()       # total duplicate rows
df.duplicated(subset=['a', 'b']).sum()  # duplicates on subset
```

---

## 6. Selection & Indexing

### Column Selection

```python
df['col']                   # Series
df[['col1', 'col2']]        # DataFrame (double brackets!)
df.col                      # attribute access (avoid — breaks with special chars)

# Select by dtype
df.select_dtypes(include='number')
df.select_dtypes(include=['object', 'category'])
df.select_dtypes(exclude=['datetime64'])
```

### Row Selection

```python
df[2:5]                     # slice by position (rows 2, 3, 4)
df[df['age'] > 30]          # boolean mask

# .loc — label-based (inclusive on both ends)
df.loc[0]                   # row with label 0
df.loc[0:5]                 # rows label 0 to 5 inclusive
df.loc[0:5, 'name':'score'] # rows 0-5, cols name to score
df.loc[:, 'name']           # all rows, col 'name'
df.loc[[1, 3, 5], ['a', 'b']]  # specific rows & cols

# .iloc — integer position-based (exclusive end)
df.iloc[0]                  # first row
df.iloc[0:5]                # rows 0, 1, 2, 3, 4
df.iloc[0:5, 0:3]           # first 5 rows, first 3 cols
df.iloc[-1]                 # last row
df.iloc[:, -1]              # last column

# .at / .iat — single value (fast scalar access)
df.at[0, 'name']            # label-based scalar
df.iat[0, 1]                # position-based scalar

# .xs — cross-section (great for MultiIndex)
df.xs('2024', level='year')
```

### Setting Values

```python
df.loc[0, 'score'] = 95.0
df.loc[df['age'] > 30, 'category'] = 'senior'
df.iloc[0:3, 2] = np.nan
```

---

## 7. Filtering & Querying

### Boolean Indexing

```python
mask = df['age'] > 25
df[mask]

# Compound conditions — use & | ~ (not and/or/not)
df[(df['age'] > 25) & (df['score'] >= 80)]
df[(df['city'] == 'NYC') | (df['city'] == 'LA')]
df[~df['name'].str.startswith('A')]

# isin
df[df['status'].isin(['active', 'pending'])]
df[~df['id'].isin(exclude_ids)]

# between (inclusive by default)
df[df['age'].between(25, 35)]
df[df['date'].between('2024-01-01', '2024-12-31')]
```

### .query() — SQL-like filtering

```python
df.query("age > 25 and score >= 80")
df.query("city in ['NYC', 'LA']")
df.query("age.between(25, 35)")      # pandas 1.5+

# Reference Python variables with @
threshold = 80
df.query("score >= @threshold")
df.query("name not in @exclude_list")

# Column names with spaces — use backticks
df.query("`first name` == 'Alice'")
```

### Comparison Methods

```python
df['col'].eq(5)     # ==
df['col'].ne(5)     # !=
df['col'].lt(5)     # <
df['col'].le(5)     # <=
df['col'].gt(5)     # >
df['col'].ge(5)     # >=
```

---

## 8. Data Cleaning

### Handling Missing Values

```python
# Detect
df.isnull()               # boolean DataFrame
df.isna()                 # alias for isnull
df.notna()

# Drop
df.dropna()                             # drop rows with ANY NaN
df.dropna(axis=1)                       # drop columns with ANY NaN
df.dropna(how='all')                    # drop rows where ALL values are NaN
df.dropna(subset=['a', 'b'])            # only consider these columns
df.dropna(thresh=3)                     # keep rows with at least 3 non-NaN

# Fill
df.fillna(0)                            # fill all NaN with 0
df.fillna({'a': 0, 'b': 'unknown'})     # column-specific fill values
df['col'].fillna(df['col'].mean())       # fill with mean
df['col'].fillna(df['col'].median())     # fill with median
df['col'].fillna(method='ffill')         # forward fill (deprecated in 2.x)
df['col'].ffill()                        # forward fill — pandas 2.x preferred
df['col'].bfill()                        # backward fill — pandas 2.x preferred
df.fillna(method='pad', limit=2)         # max 2 consecutive fills

# Interpolate
df['col'].interpolate(method='linear')
df['col'].interpolate(method='time')     # for time-indexed data
df['col'].interpolate(method='polynomial', order=2)

# Replace specific values
df.replace(-999, np.nan)
df.replace({'status': {'Y': True, 'N': False}})
df.replace([0, -1], np.nan)
df['col'].replace(r'^\s*$', np.nan, regex=True)  # whitespace-only → NaN
```

### Handling Duplicates

```python
df.duplicated()                         # boolean Series
df.duplicated(subset=['a', 'b'])        # based on subset
df.duplicated(keep='first')             # mark all but first as duplicate
df.duplicated(keep='last')
df.duplicated(keep=False)               # mark all duplicates (incl. first)

df.drop_duplicates()
df.drop_duplicates(subset=['email'])    # unique emails only
df.drop_duplicates(keep='last')         # keep most recent
df.drop_duplicates(inplace=True)        # modify in-place (use carefully)
```

### Renaming

```python
df.rename(columns={'old': 'new', 'a': 'b'})
df.rename(columns=str.upper)            # apply function to all names
df.rename(index={0: 'first'})

# Bulk clean column names
df.columns = df.columns.str.lower().str.replace(' ', '_').str.strip()
```

### Dropping Columns/Rows

```python
df.drop('col', axis=1)                  # drop column
df.drop(['col1', 'col2'], axis=1)
df.drop(columns=['col1', 'col2'])       # preferred pandas 0.21+
df.drop(0, axis=0)                      # drop row with label 0
df.drop(index=[0, 1, 2])               # drop multiple rows
df.drop(df[df['score'] < 0].index)     # conditional row drop
```

### Resetting Index

```python
df.reset_index()                        # old index becomes a column
df.reset_index(drop=True)              # discard old index
df.set_index('id')                      # use column as index
df.set_index(['year', 'month'])         # MultiIndex
```

---

## 9. Data Types & Casting

```python
df.dtypes                               # view all dtypes

# Explicit casting
df['age']   = df['age'].astype('int32')
df['score'] = df['score'].astype('float32')
df['name']  = df['name'].astype('string')   # pandas StringDtype (nullable)
df['flag']  = df['flag'].astype('boolean')  # pandas BooleanDtype (nullable)
df['cat']   = df['cat'].astype('category')

# Safer numeric conversion
pd.to_numeric(df['col'])                # raises on error
pd.to_numeric(df['col'], errors='coerce')   # bad values → NaN
pd.to_numeric(df['col'], errors='ignore')   # bad values unchanged
pd.to_numeric(df['col'], downcast='integer')  # smallest fitting int type

# Date parsing
pd.to_datetime(df['date'])
pd.to_datetime(df['date'], format='%Y-%m-%d')
pd.to_datetime(df['date'], errors='coerce')
pd.to_datetime({'year': df['yr'], 'month': df['mo'], 'day': df['dy']})

# Infer better dtypes (pandas 2.x)
df = df.convert_dtypes()   # upgrades to nullable dtypes automatically

# Memory-efficient dtype selection guide
# int64 → int32 or int16 if range allows
# float64 → float32 (lose a bit of precision)
# object string → category (if low cardinality, < 50% unique)
```

---

## 10. String Operations (str accessor)

> All methods return NaN for NaN inputs by default.

```python
s = df['name']

# Case
s.str.lower()
s.str.upper()
s.str.title()
s.str.capitalize()
s.str.swapcase()

# Stripping whitespace
s.str.strip()           # both sides
s.str.lstrip()
s.str.rstrip()
s.str.strip('$')        # strip specific chars

# Padding & alignment
s.str.pad(10, side='right', fillchar=' ')
s.str.zfill(5)          # left-pad with zeros
s.str.center(10, '-')

# Checking
s.str.startswith('A')
s.str.endswith('.csv')
s.str.contains('hello')
s.str.contains('hello', case=False, na=False)  # case-insensitive, safe for NaN
s.str.match(r'^\d+$')   # regex match from start
s.str.fullmatch(r'\d+') # entire string must match

# Search & Extract
s.str.find('o')         # position of first occurrence (-1 if not found)
s.str.count('a')        # count occurrences of substring/pattern
s.str.extract(r'(\d+)')                  # first capture group → Series
s.str.extract(r'(?P<code>\d+)-(?P<name>\w+)')  # named groups → DataFrame
s.str.extractall(r'(\d+)')              # all matches → MultiIndex DataFrame
s.str.findall(r'\d+')                   # all matches → list

# Replace
s.str.replace('old', 'new')
s.str.replace(r'\s+', ' ', regex=True)  # collapse whitespace
s.str.replace(r'\D', '', regex=True)    # keep digits only

# Splitting & Joining
s.str.split(',')                         # → list per element
s.str.split(',', expand=True)            # → DataFrame of columns
s.str.split(',', n=1, expand=True)       # max 1 split
s.str.rsplit('/', n=1, expand=True)      # split from right
s.str.join('-')                          # join list elements (if col contains lists)

# Slicing
s.str[0]                # first character
s.str[0:3]              # substring slice
s.str[-4:]              # last 4 chars

# Encoding / Bytes
s.str.encode('utf-8')
s.str.decode('utf-8')

# Misc
s.str.len()             # string length
s.str.cat(sep=', ')     # concatenate all strings into one
s.str.cat(df['col2'], sep='-')  # concatenate two columns element-wise
s.str.get_dummies(sep='|')      # one-hot encode pipe-delimited values
s.str.normalize('NFC')          # Unicode normalization
```

---

## 11. DateTime Operations (dt accessor)

```python
# Parse dates
df['date'] = pd.to_datetime(df['date'])

# Attributes
df['date'].dt.year
df['date'].dt.month
df['date'].dt.day
df['date'].dt.hour
df['date'].dt.minute
df['date'].dt.second
df['date'].dt.microsecond
df['date'].dt.nanosecond
df['date'].dt.date           # Python date object
df['date'].dt.time           # Python time object
df['date'].dt.dayofweek      # 0=Mon … 6=Sun
df['date'].dt.day_name()     # 'Monday', 'Tuesday'…
df['date'].dt.month_name()   # 'January'…
df['date'].dt.dayofyear      # 1–366
df['date'].dt.weekofyear     # deprecated — use isocalendar
df['date'].dt.isocalendar()  # DataFrame with year/week/day
df['date'].dt.quarter        # 1–4
df['date'].dt.is_month_start
df['date'].dt.is_month_end
df['date'].dt.is_year_start
df['date'].dt.is_year_end
df['date'].dt.is_leap_year

# Formatting
df['date'].dt.strftime('%Y-%m-%d')
df['date'].dt.strftime('%B %d, %Y')    # 'January 01, 2024'

# Arithmetic
df['date'] + pd.Timedelta(days=7)
df['end'] - df['start']                 # → Timedelta Series
(df['end'] - df['start']).dt.days
(df['end'] - df['start']).dt.total_seconds()

# Business day offsets
from pandas.tseries.offsets import BDay, MonthEnd, YearEnd
df['date'] + BDay(5)        # 5 business days
df['date'] + MonthEnd(1)    # next month-end

# Date ranges
pd.date_range('2024-01-01', '2024-12-31', freq='D')  # daily
pd.date_range('2024-01-01', periods=12, freq='MS')    # 12 month-starts
pd.date_range('2024-01-01', periods=52, freq='W-FRI') # weekly Fridays

# Truncating / rounding
df['date'].dt.floor('H')    # floor to hour
df['date'].dt.ceil('H')     # ceil to hour
df['date'].dt.round('H')    # round to nearest hour
# Freq aliases: 'T'=minute, 'H'=hour, 'D'=day, 'W'=week, 'M'=month

# Timezone
df['date'].dt.tz_localize('UTC')
df['date'].dt.tz_convert('US/Eastern')

# Period
df['date'].dt.to_period('M')           # monthly period
df['date'].dt.to_period('Q')           # quarterly period
```

---

## 12. Sorting & Ranking

```python
# Sort by values
df.sort_values('col')
df.sort_values('col', ascending=False)
df.sort_values(['col1', 'col2'], ascending=[True, False])
df.sort_values('col', na_position='first')  # NaN at top

# Sort by index
df.sort_index()
df.sort_index(ascending=False)
df.sort_index(axis=1)           # sort columns alphabetically

# Ranking
df['rank'] = df['score'].rank()
df['rank'] = df['score'].rank(method='dense')   # no gaps
df['rank'] = df['score'].rank(method='min')     # tied → min rank
df['rank'] = df['score'].rank(ascending=False)  # rank 1 = highest
df['rank'] = df['score'].rank(pct=True)         # percentile rank

# method options: 'average', 'min', 'max', 'first', 'dense'

# nlargest / nsmallest
df.nlargest(10, 'score')
df.nsmallest(5, 'age')
df.nlargest(10, ['score', 'age'])   # multi-column tiebreak
```

---

## 13. GroupBy & Aggregation

### Basic GroupBy

```python
g = df.groupby('city')
g = df.groupby(['city', 'dept'])        # multi-column
g = df.groupby('city', sort=False)      # don't sort keys (faster)
g = df.groupby('city', observed=True)   # pandas 2.x — suppress warning for Category

# Single aggregation
g['salary'].mean()
g['salary'].sum()
g['salary'].count()
g['salary'].nunique()
g['salary'].max()
g['salary'].min()
g['salary'].std()
g['salary'].var()
g['salary'].median()
g['salary'].quantile(0.9)

# Multiple aggregations
g['salary'].agg(['mean', 'min', 'max', 'count'])

# Different aggs per column
g.agg({'salary': ['mean', 'std'], 'age': 'median', 'score': 'sum'})

# Named aggregations (pandas 0.25+)  ← clean & explicit
df.groupby('city').agg(
    avg_salary  = ('salary', 'mean'),
    max_salary  = ('salary', 'max'),
    head_count  = ('name',   'count'),
    med_age     = ('age',    'median'),
)
```

### Transform  (returns same-shape DataFrame)

```python
df['z_score'] = df.groupby('city')['salary'].transform(
    lambda x: (x - x.mean()) / x.std()
)
df['pct_of_group'] = df['salary'] / df.groupby('city')['salary'].transform('sum')
df['rank_in_group'] = df.groupby('city')['salary'].transform('rank', ascending=False)
df['cumsum'] = df.groupby('city')['salary'].transform('cumsum')
```

### Filter (keep/remove entire groups)

```python
# Keep only groups with more than 10 members
df.groupby('city').filter(lambda x: len(x) > 10)

# Keep groups where mean salary > 50000
df.groupby('dept').filter(lambda x: x['salary'].mean() > 50_000)
```

### Apply (most flexible, slowest)

```python
def top_n(x, n=3):
    return x.nlargest(n, 'salary')

df.groupby('dept').apply(top_n, n=5, include_groups=False)  # pandas 2.2+
```

### GroupBy Iteration

```python
for name, group in df.groupby('city'):
    print(name, group.shape)
```

### Resample (time-based GroupBy)

```python
df = df.set_index('date')
df.resample('D').mean()       # daily average
df.resample('W').sum()        # weekly sum
df.resample('ME').agg({'sales': 'sum', 'orders': 'count'})  # month-end
df.resample('QE').last()      # quarter-end last value
df.resample('H').ffill()      # hourly forward fill
```

---

## 14. Merging, Joining & Concatenating

### pd.concat

```python
# Stack rows (append)
pd.concat([df1, df2], ignore_index=True)
pd.concat([df1, df2], ignore_index=True, sort=False)

# Stack columns
pd.concat([df1, df2], axis=1)

# With keys (creates MultiIndex)
pd.concat([df1, df2], keys=['train', 'test'])

# Alignment
pd.concat([df1, df2], join='inner')   # only common columns
pd.concat([df1, df2], join='outer')   # all columns (default, fills NaN)
```

### pd.merge (SQL-style joins)

```python
pd.merge(left, right, on='key')                     # inner join
pd.merge(left, right, on='key', how='left')          # left join
pd.merge(left, right, on='key', how='right')         # right join
pd.merge(left, right, on='key', how='outer')         # full outer join
pd.merge(left, right, on='key', how='cross')         # cartesian product

# Different key column names
pd.merge(left, right, left_on='user_id', right_on='id')

# Multiple keys
pd.merge(left, right, on=['year', 'month'])

# Suffixes for overlapping column names
pd.merge(left, right, on='id', suffixes=('_left', '_right'))

# Validate join type (raises if violated)
pd.merge(left, right, on='id', validate='1:1')    # one-to-one
pd.merge(left, right, on='id', validate='m:1')    # many-to-one
pd.merge(left, right, on='id', validate='1:m')    # one-to-many

# Indicator column (shows which table the row came from)
pd.merge(left, right, on='id', how='outer', indicator=True)
# _merge column: 'both' | 'left_only' | 'right_only'
```

### DataFrame.join

```python
left.join(right, on='key')              # join on index by default
left.join(right, how='left')
left.join([df2, df3])                   # join multiple at once
```

### merge_asof (nearest-key merge — great for time series)

```python
# Merge on nearest key (both must be sorted)
pd.merge_asof(trades, quotes, on='timestamp',
              by='stock',
              direction='backward')   # 'backward'|'forward'|'nearest'
```

### merge_ordered

```python
pd.merge_ordered(df1, df2, on='date', fill_method='ffill')
```

---

## 15. Reshaping Data

### pivot / pivot_table

```python
# pivot — no aggregation, values must be unique per row/col combo
df.pivot(index='date', columns='product', values='sales')

# pivot_table — with aggregation (handles duplicates)
pd.pivot_table(df,
    values='sales',
    index='date',
    columns='region',
    aggfunc='sum',
    fill_value=0,
    margins=True,          # add 'All' row/col totals
    margins_name='Total'
)

# Multiple values
pd.pivot_table(df,
    values=['sales', 'qty'],
    index='region',
    columns='product',
    aggfunc={'sales': 'sum', 'qty': 'mean'}
)
```

### melt (wide → long)

```python
# Unpivot: convert columns into rows
pd.melt(df,
    id_vars=['id', 'date'],         # columns to keep
    value_vars=['Q1', 'Q2', 'Q3', 'Q4'],  # columns to unpivot
    var_name='quarter',             # name for the variable column
    value_name='revenue'            # name for the value column
)
```

### stack / unstack

```python
# stack: move columns into row index level
df.stack()                          # innermost column level → row index
df.stack(level=0)                   # specific level

# unstack: move row index level into columns
df.unstack()                        # innermost row level → columns
df.unstack(level='region')          # specific level name
df.unstack(fill_value=0)            # fill NaN
```

### crosstab

```python
pd.crosstab(df['gender'], df['dept'])
pd.crosstab(df['gender'], df['dept'], normalize='index')  # row percentages
pd.crosstab(df['gender'], df['dept'], values=df['salary'], aggfunc='mean')
```

### explode (list values → multiple rows)

```python
df = pd.DataFrame({'id': [1, 2], 'tags': [['a', 'b'], ['c']]})
df.explode('tags')              # one row per tag
df.explode('tags', ignore_index=True)
```

### get_dummies (one-hot encoding)

```python
pd.get_dummies(df['color'])
pd.get_dummies(df, columns=['color', 'size'])
pd.get_dummies(df, columns=['color'], prefix='clr', drop_first=True)
pd.get_dummies(df, columns=['color'], dtype=int)    # 0/1 ints instead of bool
```

---

## 16. Apply, Map & Transform

```python
# Series.map — element-wise mapping
df['grade'] = df['score'].map({90: 'A', 80: 'B', 70: 'C'})
df['upper'] = df['name'].map(str.upper)
df['log']   = df['value'].map(np.log)

# Series.apply — apply callable to each element
df['col'].apply(lambda x: x ** 2)
df['col'].apply(np.sqrt)

# DataFrame.apply — apply along axis
df.apply(np.sum, axis=0)            # column sums
df.apply(np.sum, axis=1)            # row sums
df.apply(lambda col: col.max() - col.min())  # range per column

# DataFrame.applymap / map (pandas 2.1+)
df.map(lambda x: round(x, 2))      # pandas 2.1+ replaces applymap
df.applymap(lambda x: round(x, 2)) # deprecated in pandas 2.1

# Vectorized operations (MUCH faster than apply)
df['tax'] = df['salary'] * 0.3                  # prefer this
df['full_name'] = df['first'] + ' ' + df['last'] # prefer this

# pipe — chain multiple operations cleanly
df.pipe(remove_outliers).pipe(normalize_scores).pipe(add_features)

def remove_outliers(df, col='score', n_std=3):
    mu, sigma = df[col].mean(), df[col].std()
    return df[df[col].between(mu - n_std*sigma, mu + n_std*sigma)]
```

---

## 17. Window Functions

### Rolling

```python
df['col'].rolling(window=7).mean()           # 7-period moving average
df['col'].rolling(window=7).sum()
df['col'].rolling(window=7).std()
df['col'].rolling(window=7, min_periods=1).mean()  # allow partial windows
df['col'].rolling(window='7D').mean()        # time-based window (requires DatetimeIndex)

# Rolling with GroupBy
df.groupby('stock')['price'].transform(lambda x: x.rolling(7).mean())
```

### Expanding

```python
df['col'].expanding().mean()    # cumulative mean
df['col'].expanding().max()     # running max
df['col'].expanding(min_periods=5).std()
```

### Exponentially Weighted (EWM)

```python
df['col'].ewm(span=10).mean()       # EMA span=10
df['col'].ewm(alpha=0.1).mean()     # explicit smoothing factor
df['col'].ewm(halflife=5).mean()    # half-life
df['col'].ewm(com=9).std()          # center of mass
```

### Cumulative Functions

```python
df['col'].cumsum()
df['col'].cumprod()
df['col'].cummax()
df['col'].cummin()
df['col'].pct_change()          # period-over-period % change
df['col'].pct_change(periods=12)  # year-over-year (monthly data)
df['col'].diff()                # first difference
df['col'].diff(periods=7)       # 7-period difference
```

---

## 18. MultiIndex

### Creating MultiIndex

```python
# From DataFrame groupby
df_multi = df.set_index(['region', 'dept'])

# From tuples
idx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['letter', 'num'])

# From product (all combinations)
idx = pd.MultiIndex.from_product([['A', 'B'], [1, 2, 3]], names=['letter', 'num'])
```

### Selecting from MultiIndex

```python
df_multi.loc['East']                       # outer level
df_multi.loc[('East', 'Engineering')]      # exact
df_multi.loc[('East', slice(None)), :]     # all inner values for 'East'

# xs for cross-section
df_multi.xs('Engineering', level='dept')
df_multi.xs(('East', 'Engineering'))

# IndexSlice for complex selection
idx = pd.IndexSlice
df_multi.loc[idx['East', 'Engineering'], :]
df_multi.loc[idx[:, 'Engineering'], 'salary']
```

### Manipulating MultiIndex

```python
df_multi.reset_index()                     # flatten to regular columns
df_multi.index.get_level_values('region') # values at one level
df_multi.swaplevel('region', 'dept')       # swap level order
df_multi.sort_index(level='region')
df_multi.droplevel('dept')                 # remove a level

# Flatten MultiIndex columns after groupby
df.columns = ['_'.join(c) for c in df.columns]
```

---

## 19. Categorical Data

> Use when a column has low cardinality (many repeats). Saves memory and speeds up groupby.

```python
df['dept'] = df['dept'].astype('category')

# All category operations
df['dept'].cat.categories              # Index of unique categories
df['dept'].cat.codes                   # integer codes
df['dept'].cat.ordered                 # is it ordered?

# Set specific categories
df['grade'] = pd.Categorical(
    df['grade'],
    categories=['F', 'D', 'C', 'B', 'A'],
    ordered=True
)

df['grade'] > 'C'   # works correctly for ordered categories

df['dept'].cat.add_categories(['NewDept'])
df['dept'].cat.remove_categories(['OldDept'])
df['dept'].cat.rename_categories({'IT': 'Information Technology'})
df['dept'].cat.reorder_categories(['HR', 'IT', 'Finance'])
df['dept'].cat.remove_unused_categories()

# CategoricalIndex — useful for groupby speed
pd.CategoricalIndex(df['dept'])
```

---

## 20. Plotting

> Pandas wraps matplotlib; use seaborn/plotly for production visuals.

```python
import matplotlib.pyplot as plt

# Line
df['price'].plot()
df[['price', 'volume']].plot(subplots=True, figsize=(10, 6))

# Bar
df.groupby('region')['sales'].sum().plot(kind='bar')
df.groupby('region')['sales'].sum().plot(kind='barh')    # horizontal

# Histogram
df['age'].plot(kind='hist', bins=20)
df['age'].plot.hist(bins=20, alpha=0.5)

# Box
df[['salary', 'bonus']].plot(kind='box')

# Scatter
df.plot.scatter(x='age', y='salary', c='score', colormap='viridis', alpha=0.5)

# Area
df[['Q1', 'Q2', 'Q3', 'Q4']].plot.area(stacked=True)

# Pie
df.groupby('dept')['count'].sum().plot.pie(autopct='%1.1f%%')

# Hexbin (great for dense data)
df.plot.hexbin(x='x', y='y', gridsize=25)

# KDE / Density
df['salary'].plot.kde()
df['salary'].plot.density()

# Common plot parameters
df['col'].plot(
    figsize=(12, 5),
    title='My Chart',
    xlabel='X Label',
    ylabel='Y Label',
    color='steelblue',
    linestyle='--',
    marker='o',
    grid=True,
    legend=True,
    fontsize=12,
    rot=45            # x-tick rotation
)
plt.tight_layout()
plt.savefig('chart.png', dpi=150, bbox_inches='tight')
plt.show()
```

---

## 21. Performance & Memory Optimization

> Critical when working with 1–3 GB datasets

### Profiling First

```python
df.info(memory_usage='deep')         # exact memory per column
df.memory_usage(deep=True).sum() / 1e6   # total MB
df.dtypes.value_counts()             # dtype distribution
```

### Dtype Downcasting

```python
def optimize_dtypes(df):
    """Typical memory reduction: 50–70%"""
    for col in df.select_dtypes(include='int64').columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    for col in df.select_dtypes(include='float64').columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include='object').columns:
        if df[col].nunique() / len(df) < 0.5:  # <50% unique → category
            df[col] = df[col].astype('category')
    return df

df = optimize_dtypes(df)
```

### Chunked Processing

```python
# Read in chunks, process, then combine
result = (
    pd.concat([
        chunk.query("status == 'active'")
             .assign(month=chunk['date'].dt.to_period('M'))
        for chunk in pd.read_csv('large.csv', chunksize=500_000)
    ], ignore_index=True)
)
```

### Avoid Common Slow Patterns

```python
# SLOW — row-by-row iteration
for idx, row in df.iterrows():        # AVOID for large data
    df.at[idx, 'new'] = row['a'] + row['b']

# FAST — vectorized
df['new'] = df['a'] + df['b']         # USE THIS

# SLOW — apply for simple math
df['new'] = df['a'].apply(lambda x: x * 2)

# FAST — vectorized
df['new'] = df['a'] * 2

# itertuples is faster than iterrows if you must loop
for row in df.itertuples(index=False):
    print(row.name, row.salary)
```

### Use eval() and query() for Large DataFrames

```python
# eval uses numexpr under the hood — faster for large arrays
df.eval('new_col = a + b * c', inplace=True)
result = df.query("a > 100 and b < 200")   # avoids creating intermediate boolean arrays
```

### Copy vs View

```python
# SettingWithCopyWarning — always chain or use .copy()
# WRONG — may not modify original
sub = df[df['age'] > 30]
sub['new'] = 1                  # SettingWithCopyWarning!

# CORRECT
sub = df[df['age'] > 30].copy()
sub['new'] = 1

# ALSO CORRECT — use .loc directly on original
df.loc[df['age'] > 30, 'new'] = 1
```

### Parquet > CSV for I/O Performance

| Format | Write 1GB | Read 1GB | Compressed | Type-safe |
|--------|-----------|----------|------------|-----------|
| CSV    | ~30s      | ~20s     | No         | No        |
| Parquet| ~5s       | ~2s      | Yes        | Yes       |
| Feather| ~3s       | ~1s      | Optional   | Yes       |

```python
# Save once as parquet, read many times
df.to_parquet('data.parquet', compression='snappy', index=False)
df = pd.read_parquet('data.parquet', columns=['id', 'sales', 'date'])
```

### PyArrow Backend (pandas 2.x+)

```python
# Use ArrowDtype for better performance and nullable types
df = pd.read_parquet('file.parquet', dtype_backend='pyarrow')
df = pd.read_csv('file.csv', dtype_backend='pyarrow')

# Convert existing DataFrame
df = df.convert_dtypes(dtype_backend='pyarrow')
```

### Useful Libraries Alongside Pandas

```python
import modin.pandas as pd          # drop-in replacement, uses Ray/Dask, faster for big data
import dask.dataframe as dd         # out-of-core parallel processing
import polars as pl                 # blazing-fast alternative (Rust-based)
import numba                        # JIT-compile python loops → C speed
```

---

## 22. Pandas Options & Settings

```python
# Display
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.2f}'.format)
pd.set_option('display.max_colwidth', 50)
pd.set_option('display.width', 1000)

# Use context manager for temporary settings
with pd.option_context('display.max_rows', 200, 'display.max_columns', 20):
    print(df)

# Mode
pd.set_option('mode.chained_assignment', None)   # silence SettingWithCopyWarning (use carefully)
pd.set_option('mode.chained_assignment', 'warn') # default
pd.set_option('mode.chained_assignment', 'raise')# treat as error

# Future behavior
pd.set_option('future.infer_string', True)  # pandas 3.x string behavior

# Reset
pd.reset_option('display.max_rows')
pd.reset_option('all')

# Get
pd.get_option('display.max_rows')
```

---

## 23. Common Patterns & Recipes

### Add/Assign Columns

```python
df['new_col'] = df['a'] + df['b']

# .assign — functional style (great for chaining, returns new DataFrame)
df = (df
    .assign(
        full_name  = lambda x: x['first'] + ' ' + x['last'],
        tax        = lambda x: x['salary'] * 0.3,
        net_salary = lambda x: x['salary'] - x['tax'],   # reference prior assignment!
        is_senior  = lambda x: x['age'] > 40
    )
)
```

### Conditional Column (np.where / np.select)

```python
# Binary condition
df['tier'] = np.where(df['score'] >= 80, 'high', 'low')

# Multiple conditions
conditions  = [df['score'] >= 90, df['score'] >= 80, df['score'] >= 70]
choices     = ['A', 'B', 'C']
df['grade'] = np.select(conditions, choices, default='F')
```

### Bin / Discretize

```python
pd.cut(df['age'], bins=4)                           # equal-width bins
pd.cut(df['age'], bins=[0, 18, 35, 60, 100], labels=['teen', 'young', 'mid', 'senior'])
pd.qcut(df['score'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])  # equal-frequency (quantile)
pd.qcut(df['score'], q=4, duplicates='drop')         # handle duplicate edges
```

### Method Chaining (the pandas way)

```python
result = (
    pd.read_csv('sales.csv', parse_dates=['date'])
      .rename(columns=str.lower)
      .query("region != 'unknown'")
      .dropna(subset=['sales', 'product'])
      .assign(
          month       = lambda x: x['date'].dt.to_period('M'),
          sales_k     = lambda x: x['sales'] / 1000,
      )
      .groupby(['month', 'product'])
      .agg(total_sales=('sales_k', 'sum'), orders=('sales_k', 'count'))
      .reset_index()
      .sort_values('total_sales', ascending=False)
)
```

### Window Rank within Group

```python
df['rank_in_dept'] = (
    df.groupby('dept')['salary']
      .rank(method='dense', ascending=False)
      .astype(int)
)
# Keep only top 3 per dept
top3 = df[df['rank_in_dept'] <= 3]
```

### Forward / Backward Fill within Group

```python
df['price'] = df.groupby('stock')['price'].ffill()
```

### Flatten Nested JSON into DataFrame

```python
import json
from pandas import json_normalize

with open('data.json') as f:
    raw = json.load(f)

df = json_normalize(raw,
    record_path='orders',
    meta=['customer_id', 'customer_name'],
    sep='_'
)
```

### Wide-to-Long with Multiple Metrics

```python
pd.wide_to_long(df,
    stubnames=['sales', 'cost'],
    i='id',
    j='quarter',
    sep='_Q'
)
# Converts: sales_Q1, sales_Q2, cost_Q1, cost_Q2 → long format
```

### Efficient Isin with Large Sets

```python
big_set = set(valid_ids)             # Python set lookup is O(1)
df[df['id'].isin(big_set)]
```

### Rolling Correlation

```python
df['rolling_corr'] = df['a'].rolling(30).corr(df['b'])
```

### Weighted Average in GroupBy

```python
wavg = lambda x: np.average(x['sales'], weights=x['units'])
df.groupby('region').apply(wavg, include_groups=False)
```

### Add Row Totals / Column Totals

```python
df.loc['Total'] = df.sum(numeric_only=True)         # row total
df['Row_Total'] = df.sum(axis=1, numeric_only=True) # column total
```

### Cross-Tabulation with Chi-Squared

```python
ct = pd.crosstab(df['gender'], df['dept'])
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(ct)
```

### Detect Outliers with IQR

```python
def flag_outliers(series):
    Q1, Q3 = series.quantile([0.25, 0.75])
    IQR    = Q3 - Q1
    return ~series.between(Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)

df['is_outlier'] = df.groupby('dept')['salary'].transform(flag_outliers)
```

### Parse Fixed-Width Files

```python
df = pd.read_fwf('fixed.txt', widths=[10, 15, 8, 12], names=['id', 'name', 'age', 'score'])
```

### Sampling Strategies

```python
df.sample(n=1000, random_state=42)                     # random sample
df.sample(frac=0.1, random_state=42)                   # 10% sample
df.sample(n=1000, weights='weight_col', random_state=42) # weighted sample

# Stratified sample — same proportion per group
df.groupby('category', group_keys=False).apply(
    lambda x: x.sample(frac=0.1, random_state=42)
)
```

### Pipe Chain with Logging

```python
def log_step(df, step_name):
    print(f"[{step_name}] shape={df.shape}")
    return df

df = (df
    .pipe(log_step, 'raw')
    .dropna(subset=['id'])
    .pipe(log_step, 'after dropna')
    .query("value > 0")
    .pipe(log_step, 'after filter')
)
```

---

## Quick Reference Card

| Task | Method |
|------|--------|
| First / last N rows | `df.head(N)` / `df.tail(N)` |
| Shape | `df.shape` |
| Column dtypes | `df.dtypes` |
| Missing count | `df.isnull().sum()` |
| Unique count per col | `df.nunique()` |
| Frequency table | `df['col'].value_counts()` |
| Summary stats | `df.describe()` |
| Select columns | `df[['a', 'b']]` |
| Filter rows | `df[df['x'] > 5]` or `df.query("x > 5")` |
| Label-based select | `df.loc[rows, cols]` |
| Position-based select | `df.iloc[rows, cols]` |
| Drop column | `df.drop(columns=['col'])` |
| Drop NA | `df.dropna(subset=['col'])` |
| Fill NA | `df['col'].fillna(0)` |
| Rename columns | `df.rename(columns={'old': 'new'})` |
| Add column | `df['new'] = ...` |
| Group & agg | `df.groupby('g').agg(x=('col','sum'))` |
| Sort | `df.sort_values('col', ascending=False)` |
| Merge | `pd.merge(df1, df2, on='key')` |
| Concat rows | `pd.concat([df1, df2], ignore_index=True)` |
| Pivot | `df.pivot_table(values, index, columns, aggfunc)` |
| Melt | `pd.melt(df, id_vars, value_vars)` |
| Apply function | `df['col'].apply(func)` |
| String ops | `df['col'].str.lower()` |
| Date parts | `df['date'].dt.year` |
| Rolling mean | `df['col'].rolling(7).mean()` |
| Save parquet | `df.to_parquet('file.parquet', index=False)` |
| Optimize memory | `df.convert_dtypes()` + `astype('category')` |

---

*Generated by a senior Python/Pandas data engineer. Covers pandas 2.x. Always profile before optimizing.*