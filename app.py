
AI DATA SCIENTIST PRO - COMPLETE COLAB WEB APPLICATION
# Upload + Sample Datasets | Advanced EDA | Graph Studio
# Deep Dataset Analyst | K-Means | PDF + CSV Reports
# ============================================================

# 1. Install compatible packages
%pip -q uninstall -y plotly
%pip -q install gradio pandas numpy "plotly==5.24.1" scikit-learn reportlab openpyxl requests

import os, re, tempfile, warnings, requests
import numpy as np
import pandas as pd
import gradio as gr
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from xml.sax.saxutils import escape

warnings.filterwarnings("ignore")

TEMP = tempfile.gettempdir()

# ============================================================
# 2. DATASET READER
# ============================================================

def read_dataset(path):
if not path or not os.path.exists(path):
raise gr.Error("Please upload a dataset first.")

ext = os.path.splitext(path)[1].lower()

if ext in [".xlsx", ".xls"]:
df = pd.read_excel(path)
else:
errors = []
df = None
for enc in ["utf-8-sig", "utf-8", "cp1252", "latin1"]:
try:
df = pd.read_csv(path, encoding=enc, sep=None,
engine="python", on_bad_lines="warn")
break
except Exception as e:
errors.append(str(e))
if df is None:
raise gr.Error("CSV read nahi ho paayi. File format check karo.")

if df.empty:
raise gr.Error("Dataset empty hai.")

if len(df.columns) < 1:
raise gr.Error("Dataset mein columns nahi mile.")

return df


# ============================================================
# 3. DATASET PROFILE + CHART RECOMMENDATIONS
# ============================================================

def profile_dataset(df):
numeric = df.select_dtypes(include=np.number).columns.tolist()
categorical = df.select_dtypes(include=["object", "category"]).columns.tolist()

date_cols = []
for col in df.columns:
if any(k in str(col).lower() for k in ["date", "time"]):
converted = pd.to_datetime(df[col], errors="coerce")
if converted.notna().mean() > 0.5:
date_cols.append(col)

recommendations = []

if numeric:
recommendations.append("Histogram: numeric column distribution")
recommendations.append("Box Plot: numeric spread and outliers")
recommendations.append("Violin Plot: distribution shape")

if categorical:
recommendations.append("Bar Chart: category frequency")
recommendations.append("Pie / Donut: category proportions")

if len(numeric) >= 2:
recommendations.append("Scatter Plot: relationship between two numeric columns")
recommendations.append("Correlation Heatmap: numeric relationships")
recommendations.append("Scatter Matrix: multiple numeric relationships")

if date_cols and numeric:
recommendations.append("Line / Area Chart: time-based trend")

if categorical and numeric:
recommendations.append("Grouped Bar: numeric comparison by category")
recommendations.append("Treemap: category contribution")

if not recommendations:
recommendations.append("Dataset has limited chartable columns.")

return numeric, categorical, date_cols, recommendations


# ============================================================
# 4. AUTOMATIC CHARTS
# ============================================================

def automatic_charts(df, numeric, categorical, dates):
charts = []

if numeric:
c = numeric[0]
charts.append(("Numeric Distribution",
px.histogram(df, x=c, nbins=35,
title=f"Distribution of {c}")))

charts.append(("Box Plot",
px.box(df, y=c, title=f"Spread and Outliers: {c}")))

if categorical:
c = categorical[0]
counts = df[c].fillna("Missing").astype(str).value_counts().head(15)
chart_df = counts.rename_axis(c).reset_index(name="Count")
charts.append(("Category Frequency",
px.bar(chart_df, x=c, y="Count",
title=f"Top Categories: {c}")))

if len(numeric) >= 2:
charts.append(("Correlation Heatmap",
px.imshow(df[numeric].corr().round(2),
text_auto=True, aspect="auto",
title="Numeric Correlation Heatmap")))

if dates and numeric:
temp = df[[dates[0], numeric[0]]].copy()
temp[dates[0]] = pd.to_datetime(temp[dates[0]], errors="coerce")
temp[numeric[0]] = pd.to_numeric(temp[numeric[0]], errors="coerce")
temp = temp.dropna().sort_values(dates[0])

if len(temp):
temp["_Period"] = temp[dates[0]].dt.to_period("M").astype(str)
trend = temp.groupby("_Period")[numeric[0]].sum().reset_index()
charts.append(("Time Trend",
px.line(trend, x="_Period", y=numeric[0],
markers=True, title="Monthly Trend")))

return charts


# ============================================================
# 5. K-MEANS MACHINE LEARNING
# ============================================================

def run_kmeans(df, numeric):
if len(numeric) < 2 or len(df) < 5:
return None, None, "K-Means ke liye kam se kam 2 numeric columns aur 5 rows chahiye."

# Avoid obvious identifier columns
features = [
c for c in numeric
if not any(word in str(c).lower()
for word in ["id", "code", "phone", "zip", "postal"])
]

if len(features) < 2:
features = numeric[:]

X = df[features].replace([np.inf, -np.inf], np.nan)
X = X.fillna(X.median()).fillna(0)

# Remove constant columns
X = X.loc[:, X.nunique() > 1]

if X.shape[1] < 2:
return None, None, "Suitable numeric features nahi mile."

# Limit memory for very large datasets
X = X.iloc[:50000].copy()

scaler = StandardScaler()
scaled = scaler.fit_transform(X)

k = min(4, len(X) - 1)
if k < 2:
return None, None, "K-Means ke liye data kam hai."

model = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = model.fit_predict(scaled)

result = df.iloc[:len(X)].copy()
result["Cluster"] = labels

summary = result.groupby("Cluster")[features].mean().round(2).reset_index()

score = None
if len(X) > k and len(np.unique(labels)) > 1:
sample_n = min(2000, len(X))
try:
score = silhouette_score(
scaled, labels, sample_size=sample_n, random_state=42
)
except Exception:
pass

fig = None
if len(features) >= 2:
fig = px.scatter(
result, x=features[0], y=features[1],
color=result["Cluster"].astype(str),
title="K-Means Customer / Data Segments"
)

message = f"K-Means completed | Clusters: {k} | Features: {len(features)}"
if score is not None:
message += f" | Silhouette Score: {score:.3f}"

return result, summary, message, fig


# ============================================================
# 6. PDF REPORT
# ============================================================

def create_pdf(df, missing, numeric_summary, ml_message, recommendations):
path = os.path.join(TEMP, "AI_Data_Scientist_Pro_Report.pdf")
styles = getSampleStyleSheet()
story = []

story.append(Paragraph("AI DATA SCIENTIST PRO - REPORT", styles["Title"]))
story.append(Spacer(1, 15))

lines = [
f"Total Rows: {len(df):,}",
f"Total Columns: {len(df.columns)}",
f"Duplicate Rows: {df.duplicated().sum():,}",
f"Missing Cells: {df.isnull().sum().sum():,}",
f"Numeric Columns: {len(df.select_dtypes(include=np.number).columns)}",
f"ML Result: {ml_message}"
]

for line in lines:
story.append(Paragraph(escape(line), styles["BodyText"]))
story.append(Spacer(1, 6))

story.append(Spacer(1, 12))
story.append(Paragraph("Data Quality - Missing Values", styles["Heading2"]))

table_data = [["Column", "Missing", "Missing %"]]
for _, row in missing.head(35).iterrows():
table_data.append([
str(row["Column"])[:30],
str(row["Missing Values"]),
str(row["Missing %"])
])

table = Table(table_data, repeatRows=1)
table.setStyle(TableStyle([
("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#263B5A")),
("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
("FONTSIZE", (0, 0), (-1, -1), 8),
("PADDING", (0, 0), (-1, -1), 5)
]))
story.append(table)

story.append(Spacer(1, 15))
story.append(Paragraph("Chart Recommendations", styles["Heading2"]))
for item in recommendations:
story.append(Paragraph("• " + escape(item), styles["BodyText"]))

if not numeric_summary.empty:
story.append(Spacer(1, 15))
story.append(Paragraph("Numeric Statistics", styles["Heading2"]))
data = [["Statistic"] + [str(c)[:15] for c in numeric_summary.columns[:6]]]
for idx, row in numeric_summary.head(8).iterrows():
data.append([str(idx)] + [str(v)[:15] for v in row.iloc[:6].values])
t = Table(data, repeatRows=1)
t.setStyle(TableStyle([
("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
("FONTSIZE", (0, 0), (-1, -1), 7),
("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey)
]))
story.append(t)

SimpleDocTemplate(path, pagesize=A4).build(story)
return path


# ============================================================
# 7. MAIN ANALYSIS FUNCTION
# ============================================================

def analyze_file(file_path):
df = read_dataset(file_path)
original = df.copy()

duplicates = int(df.duplicated().sum())
numeric, categorical, dates, recommendations = profile_dataset(df)

missing = pd.DataFrame({
"Column": df.columns.astype(str),
"Missing Values": df.isnull().sum().values,
"Missing %": (df.isnull().mean().values * 100).round(2)
})

cleaned = df.copy()
for col in cleaned.columns:
if cleaned[col].isnull().any():
if pd.api.types.is_numeric_dtype(cleaned[col]):
med = cleaned[col].median()
cleaned[col] = cleaned[col].fillna(med if pd.notna(med) else 0)
elif pd.api.types.is_datetime64_any_dtype(cleaned[col]):
pass
else:
cleaned[col] = cleaned[col].fillna("Unknown")

cleaned = cleaned.drop_duplicates()

numeric_summary = (
df[numeric].describe().round(2) if numeric else pd.DataFrame()
)

charts = automatic_charts(df, numeric, categorical, dates)
fig1 = charts[0][1] if len(charts) > 0 else None
fig2 = charts[1][1] if len(charts) > 1 else None
fig3 = charts[2][1] if len(charts) > 2 else None

cluster_data, cluster_summary, ml_message, cluster_fig = run_kmeans(df, numeric)
if cluster_data is None:
cluster_data = pd.DataFrame()

# Sales / revenue detection
lower = {str(c).lower(): c for c in df.columns}
revenue = None

if "quantity" in lower and "unitprice" in lower:
revenue = (
pd.to_numeric(df[lower["quantity"]], errors="coerce") *
pd.to_numeric(df[lower["unitprice"]], errors="coerce")
).sum()

overview = f"""
# ✨ AI DATA SCIENTIST PRO

## Dataset Overview
| Metric | Result |
|---|---:|
| Rows | {len(df):,} |
| Columns | {len(df.columns)} |
| Duplicate rows | {duplicates:,} |
| Missing cells | {int(df.isnull().sum().sum()):,} |
| Numeric columns | {len(numeric)} |
| Categorical columns | {len(categorical)} |
| Date columns | {len(dates)} |

## Revenue Analysis
{f"Total calculated revenue: **{revenue:,.2f}**" if revenue is not None else "No standard Quantity + UnitPrice columns found."}

## Machine Learning
{ml_message}

## Recommended Graphs
""" + "\n".join("- " + x for x in recommendations)

csv_path = os.path.join(TEMP, "cleaned_dataset.csv")
cleaned.to_csv(csv_path, index=False)

cluster_path = os.path.join(TEMP, "dataset_clusters.csv")
if not cluster_data.empty:
cluster_data.to_csv(cluster_path, index=False)
else:
cluster_path = None

pdf_path = create_pdf(df, missing, numeric_summary, ml_message, recommendations)

state = {
"df": df,
"cleaned": cleaned,
"numeric": numeric,
"categorical": categorical,
"dates": dates,
"revenue": revenue,
"clusters": cluster_data
}

return (
state, overview, cleaned.head(25), missing, numeric_summary,
fig1, fig2, fig3, cluster_summary, cluster_fig,
csv_path, cluster_path, pdf_path,
gr.update(choices=[str(c) for c in df.columns], value=None),
gr.update(choices=[str(c) for c in df.columns], value=None)
)


# ============================================================
# 8. ADVANCED GRAPH STUDIO
# ============================================================

CHART_TYPES = [
"Bar", "Line", "Scatter", "Histogram", "Box", "Violin",
"Pie", "Donut", "Area", "Heatmap", "Treemap", "Sunburst",
"Funnel", "Density Heatmap", "Strip", "ECDF",
"Scatter Matrix", "Parallel Coordinates"
]

def build_chart(state, chart_type, x, y):
if not state:
raise gr.Error("Pehle dataset analyze karo.")

df = state["df"].copy()
x = x if x in df.columns else None
y = y if y in df.columns else None

if chart_type == "Scatter Matrix":
cols = state["numeric"][:8]
if len(cols) < 2:
raise gr.Error("Scatter Matrix ke liye 2 numeric columns chahiye.")
fig = px.scatter_matrix(df, dimensions=cols, title="Scatter Matrix")

elif chart_type == "Parallel Coordinates":
cols = state["numeric"][:8]
if len(cols) < 2:
raise gr.Error("Parallel Coordinates ke liye 2 numeric columns chahiye.")
fig = px.parallel_coordinates(df, dimensions=cols,
title="Parallel Coordinates")

elif chart_type == "Heatmap":
if len(state["numeric"]) < 2:
raise gr.Error("Heatmap ke liye 2 numeric columns chahiye.")
fig = px.imshow(df[state["numeric"]].corr().round(2),
text_auto=True, title="Correlation Heatmap")

else:
if not x:
x = state["numeric"][0] if state["numeric"] else df.columns[0]

if chart_type in ["Pie", "Donut", "Treemap", "Sunburst", "Funnel"]:
if not y:
y = state["numeric"][0] if state["numeric"] else None

if chart_type in ["Scatter", "Line", "Area", "Density Heatmap"]:
if not y:
choices = [c for c in state["numeric"] if c != x]
y = choices[0] if choices else None

if chart_type in ["Scatter", "Line", "Area", "Density Heatmap"] and not y:
raise gr.Error("Is graph ke liye Y column bhi select karo.")

if chart_type == "Bar":
fig = px.bar(df, x=x, y=y, title=f"{y or 'Count'} by {x}")
elif chart_type == "Line":
fig = px.line(df, x=x, y=y, markers=True, title=f"{y} over {x}")
elif chart_type == "Scatter":
fig = px.scatter(df, x=x, y=y, title=f"{y} vs {x}")
elif chart_type == "Histogram":
fig = px.histogram(df, x=x, title=f"Distribution of {x}")
elif chart_type == "Box":
fig = px.box(df, x=x if y else None, y=y or x, title="Box Plot")
elif chart_type == "Violin":
fig = px.violin(df, x=x if y else None, y=y or x, box=True,
title="Violin Plot")
elif chart_type in ["Pie", "Donut"]:
counts = df[x].fillna("Missing").astype(str).value_counts().head(15)
plot_df = counts.rename_axis(x).reset_index(name="Count")
fig = px.pie(plot_df, names=x, values="Count",
hole=0.45 if chart_type == "Donut" else 0,
title=f"{chart_type}: {x}")
elif chart_type == "Area":
fig = px.area(df, x=x, y=y, title=f"{y} by {x}")
elif chart_type == "Treemap":
fig = px.treemap(df, path=[x], values=y, title="Treemap")
elif chart_type == "Sunburst":
fig = px.sunburst(df, path=[x], values=y, title="Sunburst")
elif chart_type == "Funnel":
fig = px.funnel(df, x=y, y=x, title="Funnel Chart")
elif chart_type == "Density Heatmap":
fig = px.density_heatmap(df, x=x, y=y, title="Density Heatmap")
elif chart_type == "Strip":
fig = px.strip(df, x=x if y else None, y=y or x, title="Strip Plot")
elif chart_type == "ECDF":
fig = px.ecdf(df, x=x, title=f"ECDF of {x}")
else:
raise gr.Error("Chart type not supported.")

fig.update_layout(template="plotly_dark", height=550,
margin=dict(l=30, r=30, t=70, b=30))
return fig


# ============================================================
# 9. DEEP DATASET AI ANALYST
# ============================================================

def ask_agent(question, state):
if not state:
return "⚠️ Pehle dataset upload aur analyze karo."

if not question or not question.strip():
return "Apna question type karo."

df = state["df"]
q = question.lower()
numeric = state["numeric"]
categorical = state["categorical"]

# Dataset-wide general questions
if any(w in q for w in ["overview", "summary", "describe dataset", "dataset ke baare"]):
return (
f"📊 DATASET SUMMARY\n\n"
f"Rows: {len(df):,}\nColumns: {len(df.columns)}\n"
f"Duplicates: {df.duplicated().sum():,}\n"
f"Missing cells: {df.isnull().sum().sum():,}\n"
f"Numeric columns: {numeric}\nCategorical columns: {categorical}"
)

if any(w in q for w in ["column", "features", "variables"]):
return "Dataset columns:\n" + "\n".join(
f"• {c} ({df[c].dtype})" for c in df.columns
)

if any(w in q for w in ["missing", "null", "empty"]):
m = df.isnull().sum().sort_values(ascending=False)
return "Missing values by column:\n" + m.to_string()

if "duplicate" in q:
return f"Duplicate rows: {df.duplicated().sum():,}"

if any(w in q for w in ["correlation", "correlated", "relationship"]):
if len(numeric) < 2:
return "Correlation ke liye 2 numeric columns chahiye."
corr = df[numeric].corr().round(3)
pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
return "Correlation matrix:\n" + corr.to_string() + \
"\n\nStrongest pairs:\n" + pairs.abs().sort_values(ascending=False).head(8).to_string()

if any(w in q for w in ["revenue", "sales", "turnover"]):
if state["revenue"] is not None:
return f"Total calculated revenue / sales: {state['revenue']:,.2f}"
return "Quantity aur UnitPrice columns nahi mile. Isliye standard revenue calculate nahi hua."

if any(w in q for w in ["highest", "maximum", "max", "largest", "top"]):
if any(w in q for w in ["country", "category", "product", "customer"]) and categorical:
cat = next((c for c in categorical if str(c).lower() in q), categorical[0])
counts = df[cat].fillna("Missing").astype(str).value_counts().head(10)
return f"Top values in {cat}:\n{counts.to_string()}"

if any(w in q for w in ["average", "mean", "minimum", "maximum", "median", "std"]):
matching = [c for c in numeric if str(c).lower() in q]
if matching:
c = matching[0]
s = pd.to_numeric(df[c], errors="coerce")
return (
f"📈 ANALYSIS: {c}\n"
f"Mean: {s.mean():,.3f}\nMedian: {s.median():,.3f}\n"
f"Minimum: {s.min():,.3f}\nMaximum: {s.max():,.3f}\n"
f"Standard Deviation: {s.std():,.3f}\n"
f"25th Percentile: {s.quantile(.25):,.3f}\n"
f"75th Percentile: {s.quantile(.75):,.3f}"
)

# Automatic category + numeric group analysis
matching_cat = [c for c in categorical if str(c).lower() in q]
matching_num = [c for c in numeric if str(c).lower() in q]

if matching_cat:
cat = matching_cat[0]
if matching_num:
num = matching_num[0]
result = df.groupby(cat)[num].agg(["count", "mean", "sum", "max"])
return f"{num} grouped by {cat}:\n{result.sort_values('sum', ascending=False).head(15).round(2).to_string()}"
counts = df[cat].fillna("Missing").astype(str).value_counts().head(20)
return f"Category analysis: {cat}\n{counts.to_string()}"

if matching_num:
c = matching_num[0]
s = pd.to_numeric(df[c], errors="coerce")
return (
f"Column: {c}\nCount: {s.count():,}\n"
f"Mean: {s.mean():,.3f}\nMedian: {s.median():,.3f}\n"
f"Min: {s.min():,.3f}\nMax: {s.max():,.3f}\n"
f"Std: {s.std():,.3f}"
)

if any(w in q for w in ["cluster", "segment", "kmeans"]):
clusters = state["clusters"]
if clusters is not None and not clusters.empty:
return "Cluster distribution:\n" + clusters["Cluster"].value_counts().sort_index().to_string()
return "K-Means result available nahi hai. Numeric columns check karo."

if any(w in q for w in ["outlier", "anomaly"]):
if not numeric:
return "Numeric columns nahi mile."
output = []
for c in numeric[:15]:
s = pd.to_numeric(df[c], errors="coerce").dropna()
if len(s):
q1, q3 = s.quantile(.25), s.quantile(.75)
iqr = q3 - q1
n = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()
output.append(f"{c}: {n:,} possible outliers")
return "IQR outlier screening:\n" + "\n".join(output)

if any(w in q for w in ["first", "preview", "sample rows"]):
return df.head(10).to_string(index=False)

return (
"Main uploaded dataset ke available data par analysis kar sakta hoon.\n\n"
"Try these questions:\n"
"• Give me complete dataset overview\n"
"• Which country/category has the most records?\n"
"• Show missing values\n"
"• Show correlation between features\n"
"• Find outliers in the dataset\n"
"• What is the average of [column name]?\n"
"• Show [category column] by [numeric column]\n"
"• Explain the K-Means clusters\n"
"• Show all dataset columns"
)


# ============================================================
# 10. SAMPLE DATASETS
# ============================================================

SAMPLES = {
"Iris Flower": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv",
"Titanic": "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
"Tips Restaurant": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/tips.csv",
"Penguins": "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/penguins.csv"
}

def load_sample(name):
if name not in SAMPLES:
raise gr.Error("Sample dataset select karo.")

try:
r = requests.get(SAMPLES[name], timeout=30)
r.raise_for_status()
path = os.path.join(TEMP, "sample_dataset.csv")
with open(path, "wb") as f:
f.write(r.content)
return analyze_file(path)
except Exception as e:
raise gr.Error(f"Sample dataset load nahi hua: {str(e)}")


# ============================================================
# 11. ATTRACTIVE GRADIO DASHBOARD
# ============================================================

CUSTOM_CSS = """
.gradio-container {
background: linear-gradient(135deg, #080d1b, #111b32, #101329) !important;
color: #f5f7ff !important;
}
h1, h2, h3 { color: #a9a4ff !important; }
button.primary {
background: linear-gradient(90deg, #625bff, #9b5cff) !important;
border: 0 !important;
}
.panel {
border: 1px solid #343c60 !important;
border-radius: 16px !important;
padding: 14px !important;
}
"""

with gr.Blocks(
theme=gr.themes.Soft(primary_hue="violet", secondary_hue="blue"),
css=CUSTOM_CSS,
title="AI Data Scientist Pro"
) as demo:

state = gr.State(None)

gr.Markdown("""
# 🤖 AI DATA SCIENTIST PRO
### Your all-in-one intelligent data analysis workspace

**Upload • Explore • Visualize • Machine Learning • Ask AI • Export**
""")

with gr.Row():
with gr.Column(scale=3):
file_input = gr.File(
label="📂 Upload your CSV / Excel dataset",
file_types=[".csv", ".xlsx", ".xls"],
type="filepath"
)
with gr.Column(scale=1):
analyze_btn = gr.Button("🚀 ANALYZE DATASET", variant="primary")

with gr.Row():
sample_dropdown = gr.Dropdown(
choices=list(SAMPLES.keys()),
label="Or select a sample dataset"
)
sample_btn = gr.Button("Load Sample Dataset")

overview = gr.Markdown("### Upload a dataset and click Analyze to begin.")

with gr.Tabs():

with gr.Tab("📊 Dataset Explorer"):
preview = gr.Dataframe(label="Cleaned Dataset Preview", interactive=False)
missing_table = gr.Dataframe(label="Missing Values Report")
numeric_table = gr.Dataframe(label="Statistical Summary")

with gr.Tab("📈 Automatic EDA"):
gr.Markdown("### Automatically recommended charts")
chart1 = gr.Plot(label="Automatic Chart 1")
chart2 = gr.Plot(label="Automatic Chart 2")
chart3 = gr.Plot(label="Automatic Chart 3")

with gr.Tab("🎨 Advanced Graph Studio"):
gr.Markdown("""
### Build your own chart
Choose a graph type and select columns. Some charts need both X and Y.
""")
with gr.Row():
chart_type = gr.Dropdown(
choices=CHART_TYPES,
value="Bar",
label="Chart Type"
)
x_column = gr.Dropdown(choices=[], label="X Column")
y_column = gr.Dropdown(choices=[], label="Y Column (if required)")
graph_btn = gr.Button("Generate Graph", variant="primary")
graph_output = gr.Plot(label="Your Custom Graph")

with gr.Tab("🧠 Machine Learning"):
gr.Markdown("### K-Means Clustering")
ml_summary = gr.Markdown("Run analysis to see ML results.")
cluster_table = gr.Dataframe(label="Cluster Summary")
cluster_plot = gr.Plot(label="Cluster Visualization")
cluster_download = gr.File(label="Download Cluster Dataset")

with gr.Tab("💬 Deep Dataset AI Analyst"):
gr.Markdown("""
### Ask questions about your uploaded dataset

The built-in analyst uses your dataset directly with Pandas and statistical analysis.
It does not send your dataset to an external AI service.
""")
question = gr.Textbox(
label="Ask your dataset",
placeholder="Example: Show correlation between features",
lines=2
)
ask_btn = gr.Button("Ask AI Analyst", variant="primary")
answer = gr.Textbox(label="Analysis Answer", lines=12)
ask_btn.click(ask_agent, inputs=[question, state], outputs=answer)
question.submit(ask_agent, inputs=[question, state], outputs=answer)

with gr.Tab("📥 Reports & Downloads"):
csv_download = gr.File(label="Download Cleaned CSV")
pdf_download = gr.File(label="Download PDF Report")

analyze_outputs = [
state, overview, preview, missing_table, numeric_table,
chart1, chart2, chart3, cluster_table, cluster_plot,
csv_download, cluster_download, pdf_download,
x_column, y_column
]

analyze_btn.click(
analyze_file,
inputs=file_input,
outputs=analyze_outputs
)

sample_btn.click(
load_sample,
inputs=sample_dropdown,
outputs=analyze_outputs
)

graph_btn.click(
build_chart,
inputs=[state, chart_type, x_column, y_column],
outputs=graph_output
)

gr.Markdown("""
---
**AI Data Scientist Pro | Portfolio Project**

Upload datasets that you have permission to use.
For very large files, processing time and memory depend on the Colab runtime.
""")

print("Starting AI Data Scientist Pro...")
demo.queue().launch(share=True, debug=False)






































