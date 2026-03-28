### 1. LIBRARY IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # FIXED: Changed 'import sns' to 'import seaborn as sns'
import plotly.express as px
import plotly.graph_objects as go

# Machine Learning utilities
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

### 2. DATA LOADING & INITIAL INSPECTION
vg = pd.read_csv("Video Games Sales (1980-2024) - Raw.csv")

# Note: In a script, .head() doesn't show anything unless you wrap it in print()
print(vg.head())
print(vg.info())

### 3. DATA CLEANING & PREPROCESSING
vg.dropna(subset=["total_sales"], inplace=True)
vg.drop(columns=["na_sales", "jp_sales", "pal_sales", "other_sales"], inplace=True)

vg['developer'] = vg['developer'].fillna('Unknown')
vg['critic_score'] = vg['critic_score'].fillna(vg['critic_score'].median())

# FIXED: Added dayfirst=True to stop the Warning
vg['release_date'] = pd.to_datetime(vg['release_date'], dayfirst=True, errors='coerce')
vg['release_year'] = vg['release_date'].dt.year

vg.drop(columns=['img', 'last_update', 'release_date'], inplace=True)
vg.dropna(subset=['release_year'], inplace=True)

### 4. EXPLORATORY DATA ANALYSIS (EDA) - VISUALIZATION
genre_sales = vg.groupby('genre')['total_sales'].sum().sort_values(ascending=False).reset_index()

fig = px.bar(genre_sales, x='genre', y='total_sales',
             title='Total Sales by Genre',
             labels={'total_sales': 'Total Sales (Millions)'},
             color='total_sales', color_continuous_scale='Viridis')
# FIXED: Removed renderer="iframe" to use the default browser
fig.show()

# Time-Series Analysis
yearly_sales = vg.groupby('release_year')['total_sales'].sum().reset_index()

plt.figure(figsize=(14, 6))
sns.lineplot(data=yearly_sales, x='release_year', y='total_sales', marker='o', color='red')
plt.title('Evolution of Video Game Sales (1980 - 2024)')
plt.ylabel('Total Sales (Millions)')
plt.show()

# Platform Trends
major_consoles = vg['console'].value_counts().head(10).index
console_trends = vg[vg['console'].isin(major_consoles)].groupby(['release_year', 'console'])['total_sales'].sum().reset_index()

fig1 = px.line(console_trends, x="release_year", y="total_sales", color="console",
               title="Annual Total Sales by Console",
               labels={"total_sales": "Sales (Millions)", "release_year": "Year"},
               markers=True)
fig1.update_layout(hovermode="x unified")
fig1.show() # FIXED: Removed renderer="iframe"

# Market Share Analysis
pub_market = vg.groupby(['publisher', 'genre'])['total_sales'].sum().reset_index()
top_pub_list = vg.groupby('publisher')['total_sales'].sum().sort_values(ascending=False).head(20).index
pub_market = pub_market[pub_market['publisher'].isin(top_pub_list)]

fig2 = px.treemap(pub_market, path=['publisher', 'genre'], values='total_sales',
                  color='total_sales', color_continuous_scale='blues',
                  title="Publisher Market Share & Genre Specialization")
fig2.show() # FIXED: Removed renderer="iframe"

# Heatmap
plt.figure(figsize=(14, 10))
pivot_table = vg[vg['console'].isin(major_consoles)].pivot_table(index='genre', columns='console', values='title', aggfunc='count').fillna(0)

sns.heatmap(pivot_table, annot=True, fmt='g', cmap='YlGnBu')
plt.xlabel('Console')
plt.ylabel('Genre')
plt.title('Game Count Heatmap: Genre vs Console Platforms', fontsize=15)
plt.show()