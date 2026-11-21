"""
SALE PROJECT - Sales Data Analysis
Uses only: numpy and pandas  (no os, no machine learning)

Put this file in the same folder as cleaned_sales.csv and run:
    python sale_project.py

It creates:
    sale_report.txt            -> full readable report
    sale_summary_tables.csv    -> region/rep/product/etc. tables in one file
    sale_monthly_trend.csv     -> month by month results
    sale_enriched_data.csv     -> your data plus new columns
"""

import numpy as np
import pandas as pd

DATA_FILE = "cleaned_sales.csv"
SLOW_SHIP_DAYS = 6      # shipments taking this many days or more are "slow"
LOW_MARGIN_PCT = 30     # orders under this profit margin are "low margin"

report = []   # report lines collected here


def add(text=""):
    report.append(text)
    print(text)


def heading(title):
    add()
    add("=" * 70)
    add(title.upper())
    add("=" * 70)


def pct(part, whole):
    """Percentage that safely returns 0 when whole is 0."""
    return np.where(whole == 0, 0, part / np.where(whole == 0, 1, whole) * 100)


# ---------------------------------------------------------------------------
# 1. LOAD DATA AND ADD NEW COLUMNS
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)
df["Order_Date"] = pd.to_datetime(df["Order_Date"])
df["Ship_Date"] = pd.to_datetime(df["Ship_Date"])

df["Month"] = df["Order_Date"].dt.to_period("M").astype(str)
df["Quarter"] = "Q" + df["Order_Date"].dt.quarter.astype(str)
df["Weekday"] = df["Order_Date"].dt.day_name()
df["Ship_Days"] = (df["Ship_Date"] - df["Order_Date"]).dt.days
df["Discount_Amount"] = df["Gross_Sales"] - df["Net_Revenue"]
df["Slow_Shipment"] = np.where(df["Ship_Days"] >= SLOW_SHIP_DAYS, "Yes", "No")
df["Margin_Level"] = np.select(
    [df["Profit_Margin_Pct"] < LOW_MARGIN_PCT, df["Profit_Margin_Pct"] < 50],
    ["Low", "Medium"],
    "High",
)

# ---------------------------------------------------------------------------
# 2. OVERALL NUMBERS
# ---------------------------------------------------------------------------
heading("Overall summary")
total_net = df["Net_Revenue"].sum()
total_profit = df["Profit"].sum()
add(f"Period                  : {df['Order_Date'].min():%d %b %Y} to {df['Order_Date'].max():%d %b %Y}")
add(f"Total orders            : {len(df)}")
add(f"Total units sold        : {df['Quantity'].sum():,}")
add(f"Gross sales             : {df['Gross_Sales'].sum():,.2f}")
add(f"Discounts given         : {df['Discount_Amount'].sum():,.2f}")
add(f"Net revenue             : {total_net:,.2f}")
add(f"Total cost              : {df['Total_Cost'].sum():,.2f}")
add(f"Total profit            : {total_profit:,.2f}")
add(f"Overall profit margin   : {total_profit / total_net * 100:.2f}%")
add(f"Average order value     : {df['Net_Revenue'].mean():,.2f}")
add(f"Average shipping days   : {df['Ship_Days'].mean():.2f}")
add(f"Slow shipments          : {(df['Slow_Shipment'] == 'Yes').sum()} orders")


# ---------------------------------------------------------------------------
# 3. PERFORMANCE BY GROUP (region, rep, product, ...)
# ---------------------------------------------------------------------------
def summarize(column):
    t = df.groupby(column).agg(
        Orders=("Order_ID", "count"),
        Units=("Quantity", "sum"),
        Net_Revenue=("Net_Revenue", "sum"),
        Profit=("Profit", "sum"),
    ).reset_index()
    t["Margin_Pct"] = pct(t["Profit"], t["Net_Revenue"])
    t["Revenue_Share_Pct"] = pct(t["Net_Revenue"], t["Net_Revenue"].sum())
    t["Avg_Order_Value"] = t["Net_Revenue"] / t["Orders"]
    t = t.sort_values("Net_Revenue", ascending=False).reset_index(drop=True)
    t.insert(0, "Rank", np.arange(1, len(t) + 1))
    return t.round(2)


group_columns = [
    ("Region", "Region"),
    ("Sales_Rep", "Sales rep"),
    ("Customer_Segment", "Customer segment"),
    ("Sales_Channel", "Sales channel"),
    ("Product_Category", "Product category"),
    ("Product_Name", "Product"),
    ("Payment_Method", "Payment method"),
]

summary_tables = []
for col, label in group_columns:
    table = summarize(col)
    heading(f"Performance by {label}")
    add(table.to_string(index=False))
    saved = table.rename(columns={col: "Name"})
    saved.insert(0, "Group_By", label)
    summary_tables.append(saved)

# ---------------------------------------------------------------------------
# 4. MONTHLY AND QUARTERLY TREND
# ---------------------------------------------------------------------------
monthly = df.groupby("Month").agg(
    Orders=("Order_ID", "count"),
    Net_Revenue=("Net_Revenue", "sum"),
    Profit=("Profit", "sum"),
).reset_index()
monthly["Margin_Pct"] = pct(monthly["Profit"], monthly["Net_Revenue"])
monthly["Revenue_Growth_Pct"] = monthly["Net_Revenue"].pct_change() * 100
monthly["Revenue_3_Month_Avg"] = monthly["Net_Revenue"].rolling(3, min_periods=1).mean()
monthly["Cumulative_Revenue"] = np.cumsum(monthly["Net_Revenue"])
monthly = monthly.round(2)

heading("Monthly trend")
add(monthly.to_string(index=False))

quarterly = df.groupby("Quarter").agg(
    Orders=("Order_ID", "count"),
    Net_Revenue=("Net_Revenue", "sum"),
    Profit=("Profit", "sum"),
).reset_index()
quarterly["Margin_Pct"] = pct(quarterly["Profit"], quarterly["Net_Revenue"])
heading("Quarterly trend")
add(quarterly.round(2).to_string(index=False))

weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
weekday = (
    df.groupby("Weekday")
    .agg(Orders=("Order_ID", "count"), Net_Revenue=("Net_Revenue", "sum"))
    .reindex(weekday_order)
    .fillna(0)
    .reset_index()
)
heading("Orders by weekday")
add(weekday.round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# 5. CROSS TABLES
# ---------------------------------------------------------------------------
heading("Net revenue: Region x Product category")
add(pd.pivot_table(df, index="Region", columns="Product_Category", values="Net_Revenue",
                   aggfunc="sum", fill_value=0, margins=True, margins_name="Total").round(2).to_string())

heading("Net revenue: Customer segment x Sales channel")
add(pd.pivot_table(df, index="Customer_Segment", columns="Sales_Channel", values="Net_Revenue",
                   aggfunc="sum", fill_value=0, margins=True, margins_name="Total").round(2).to_string())

# ---------------------------------------------------------------------------
# 6. DISCOUNT ANALYSIS
# ---------------------------------------------------------------------------
df["Discount_Pct"] = (df["Discount_Rate"] * 100).round().astype(int)
discount = df.groupby("Discount_Pct").agg(
    Orders=("Order_ID", "count"),
    Avg_Quantity=("Quantity", "mean"),
    Net_Revenue=("Net_Revenue", "sum"),
    Discount_Given=("Discount_Amount", "sum"),
    Profit=("Profit", "sum"),
).reset_index()
discount["Margin_Pct"] = pct(discount["Profit"], discount["Net_Revenue"])
heading("Discount analysis")
add(discount.round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. SHIPPING ANALYSIS
# ---------------------------------------------------------------------------
shipping = df.groupby("Sales_Channel").agg(
    Orders=("Order_ID", "count"),
    Avg_Ship_Days=("Ship_Days", "mean"),
    Max_Ship_Days=("Ship_Days", "max"),
    Slow_Shipments=("Slow_Shipment", lambda s: (s == "Yes").sum()),
).reset_index()
shipping["Slow_Shipment_Pct"] = pct(shipping["Slow_Shipments"], shipping["Orders"])
heading("Shipping by sales channel")
add(shipping.round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# 8. PRODUCT ABC RANKING (which products bring 80% of revenue)
# ---------------------------------------------------------------------------
abc = df.groupby("Product_Name").agg(
    Net_Revenue=("Net_Revenue", "sum"), Profit=("Profit", "sum")
).reset_index().sort_values("Net_Revenue", ascending=False).reset_index(drop=True)
abc["Revenue_Share_Pct"] = pct(abc["Net_Revenue"], abc["Net_Revenue"].sum())
abc["Cumulative_Share_Pct"] = np.cumsum(abc["Revenue_Share_Pct"])
before = abc["Cumulative_Share_Pct"] - abc["Revenue_Share_Pct"]
abc["Class"] = np.select([before < 80, before < 95], ["A", "B"], "C")
heading("Product ABC ranking (A = first 80% of revenue)")
add(abc.round(2).to_string(index=False))

# ---------------------------------------------------------------------------
# 9. SALES REP RANKING
# ---------------------------------------------------------------------------
reps = summarize("Sales_Rep")
reps["Performance"] = np.select(
    [reps["Rank"] <= 3, reps["Rank"] > len(reps) - 3],
    ["Top 3", "Bottom 3"],
    "Middle",
)
heading("Sales rep ranking")
add(reps[["Rank", "Sales_Rep", "Orders", "Net_Revenue", "Profit", "Margin_Pct", "Performance"]]
    .to_string(index=False))

# ---------------------------------------------------------------------------
# 10. TOP ORDERS AND LOW MARGIN ORDERS
# ---------------------------------------------------------------------------
order_cols = ["Order_ID", "Region", "Sales_Rep", "Product_Name", "Quantity",
              "Net_Revenue", "Profit", "Profit_Margin_Pct"]
heading("Top 10 orders by profit")
add(df.sort_values("Profit", ascending=False).head(10)[order_cols].to_string(index=False))

low = df[df["Margin_Level"] == "Low"].sort_values("Profit_Margin_Pct")
heading(f"Low margin orders (below {LOW_MARGIN_PCT}%)")
add(low[order_cols].to_string(index=False) if len(low) else "None found.")

# ---------------------------------------------------------------------------
# 11. KEY FINDINGS (written automatically from the numbers)
# ---------------------------------------------------------------------------
heading("Key findings")
best = lambda t, col: t.loc[t[col].idxmax()]
region_t, cat_t = summarize("Region"), summarize("Product_Category")
chan_t, seg_t = summarize("Sales_Channel"), summarize("Customer_Segment")
prod_t = summarize("Product_Name")

findings = [
    f"Best region: {region_t.iloc[0]['Region']} ({region_t.iloc[0]['Revenue_Share_Pct']:.1f}% of revenue); "
    f"lowest: {region_t.iloc[-1]['Region']}.",
    f"Best category by revenue: {cat_t.iloc[0]['Product_Category']}; "
    f"best margin: {best(cat_t, 'Margin_Pct')['Product_Category']} ({best(cat_t, 'Margin_Pct')['Margin_Pct']:.1f}%).",
    f"Top channel: {chan_t.iloc[0]['Sales_Channel']} ({chan_t.iloc[0]['Revenue_Share_Pct']:.1f}% of revenue).",
    f"Top customer segment: {seg_t.iloc[0]['Customer_Segment']}.",
    f"Top sales rep: {reps.iloc[0]['Sales_Rep']} with {reps.iloc[0]['Net_Revenue']:,.2f} net revenue.",
    f"Best product by revenue: {prod_t.iloc[0]['Product_Name']}; "
    f"most profitable: {best(prod_t, 'Profit')['Product_Name']}.",
    f"Best month: {monthly.loc[monthly['Net_Revenue'].idxmax(), 'Month']}.",
    f"{(abc['Class'] == 'A').sum()} of {len(abc)} products make up {abc.loc[abc['Class'] == 'A', 'Revenue_Share_Pct'].sum():.1f}% of revenue.",
    f"Discounts total {df['Discount_Amount'].sum():,.2f} "
    f"({df['Discount_Amount'].sum() / df['Gross_Sales'].sum() * 100:.1f}% of gross sales).",
    f"{(df['Slow_Shipment'] == 'Yes').mean() * 100:.1f}% of orders took {SLOW_SHIP_DAYS}+ days to ship.",
    f"{len(low)} orders have a margin below {LOW_MARGIN_PCT}%.",
]
last_day = df["Order_Date"].max()
if last_day != last_day + pd.offsets.MonthEnd(0):
    findings.append(f"Note: data ends on {last_day:%d %b %Y}, so the last month is incomplete.")
for i, text in enumerate(findings, 1):
    add(f"{i}. {text}")

# ---------------------------------------------------------------------------
# 12. SAVE RESULTS
# ---------------------------------------------------------------------------
pd.concat(summary_tables, ignore_index=True).to_csv("sale_summary_tables.csv", index=False)
monthly.to_csv("sale_monthly_trend.csv", index=False)

enriched = df.copy()
enriched["Order_Date"] = enriched["Order_Date"].dt.strftime("%Y-%m-%d")
enriched["Ship_Date"] = enriched["Ship_Date"].dt.strftime("%Y-%m-%d")
enriched.to_csv("sale_enriched_data.csv", index=False)

with open("sale_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(report))

print("\nSaved: sale_report.txt, sale_summary_tables.csv, sale_monthly_trend.csv, sale_enriched_data.csv")
