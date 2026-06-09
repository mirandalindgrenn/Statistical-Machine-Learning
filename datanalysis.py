import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('training_data_vt2025.csv', dtype={'ID': str}).dropna().reset_index(drop=True)

# Map categorical values to numerical values
data['increase_stock'] = data['increase_stock'].map({'low_bike_demand': 0, 'high_bike_demand': 1})

# Add humidity column with float values
data['humidity'] = data['humidity'].astype(float)

# Define pastel color palette
pastel_palette = sns.color_palette("pastel")

# Figure 1: Time-based plots
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 6))

# Hourly counts
hourly_counts_all = data[data['increase_stock'] == 1]['hour_of_day'].value_counts().sort_index()
all_hours_counts = hourly_counts_all.reindex(range(24), fill_value=0)
sns.barplot(ax=axes1[0], x=all_hours_counts.index, y=all_hours_counts.values, palette=pastel_palette)
axes1[0].set_title('High Demand by Hour of Day')
axes1[0].set_xlabel('Hour of the Day')
axes1[0].set_ylabel('Number of Data Points')

# Day of the week counts
increase_stock_1 = data[data['increase_stock'] == 1].dropna(subset=['day_of_week'])
day_of_week_counts = increase_stock_1['day_of_week'].value_counts().sort_index()
sns.barplot(ax=axes1[1], x=day_of_week_counts.index, y=day_of_week_counts.values, palette=pastel_palette)
axes1[1].set_title('High Demand by Day of Week')
axes1[1].set_xlabel('Day of the Week')
axes1[1].set_ylabel('Number of Data Points')

# Month counts
month_counts = data[data['increase_stock'] == 1]['month'].value_counts().sort_index()
sns.barplot(ax=axes1[2], x=month_counts.index, y=month_counts.values, palette=pastel_palette)
axes1[2].set_title('High Demand by Month')
axes1[2].set_xlabel('Month')
axes1[2].set_ylabel('Number of Data Points')

plt.tight_layout()
plt.savefig("figure1_time_analysis.png", dpi=300)

# Figure 2: Holiday demand
fig2, ax2 = plt.subplots(figsize=(8, 6))

# Group and count for holidays and weekdays
holiday_counts = data.groupby(['holiday', 'increase_stock']).size().unstack(fill_value=0)

# Convert to long format for plotting
holiday_demand = holiday_counts.stack().reset_index(name='Count')
holiday_demand['Type'] = holiday_demand['holiday'].map({0: 'Weekday', 1: 'Holiday'})
holiday_demand['Demand'] = holiday_demand['increase_stock'].map({0: 'Low Demand', 1: 'High Demand'})


sns.barplot(ax=ax2, x='Type', y='Count', hue='Demand', data=holiday_demand, palette=pastel_palette[:2])
ax2.set_title('Demand on Holidays and Weekdays')
ax2.set_xlabel('Day Type')
ax2.set_ylabel('Number of Data Points')

plt.tight_layout()
plt.savefig("figure2_holiday_demand.png", dpi=300)


# Figure 3: Weather/Environmental factors
fig3, axes3 = plt.subplots(2, 5, figsize=(25, 10))

features = ['precip', 'summertime', 'temp', 'dew', 'humidity', 'snowdepth', 'windspeed', 'cloudcover', 'visibility', 'snow']

for i, feature in enumerate(features):
    row = i // 5
    col = i % 5
    high_demand_data = data[data['increase_stock'] == 1] #filter for high demand
    if data[feature].count() < 5:  # Check for less than 5 non-null values
        axes3[row, col].set_title(f'Distribution of {feature} (No Data)') # Or just skip the plotting entirely
        continue # Skip to the next feature
    sns.histplot(ax=axes3[row, col], x=high_demand_data[feature], kde=True, color=pastel_palette[i % len(pastel_palette)]) #plot for high demand only
    axes3[row, col].set_title(f'Distribution of {feature}') #updated title
    axes3[row, col].set_xlabel(feature)
    axes3[row, col].set_ylabel('Frequency')

plt.tight_layout()
plt.savefig("figure3_weather_analysis.png", dpi=300)

# Print demand counts
high_demand_count = data[data['increase_stock'] == 1]['increase_stock'].value_counts()
low_demand_count = data[data['increase_stock'] == 0]['increase_stock'].value_counts()

print(f"Increase stock, high demand:\n{high_demand_count}\n")
print(f"Increase stock, low demand:\n{low_demand_count}")