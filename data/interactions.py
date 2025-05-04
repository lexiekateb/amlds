import pandas as pd
# copy and paste the below to make dataframes with the csvs

ab_count = pd.read_csv('ab_counts.csv')
ab_props = pd.read_csv('ab_props.csv')

ispal_count = pd.read_csv('ispal_counts.csv')
ispal_props = pd.read_csv('ispal_props.csv')

# treat the days as "iters". Indexed by day

# to get the data from day 10:
ab_day_10 = ab_count[ab_count['day'] == 10]
ab_day_10_props = ab_props[ab_props['day'] == 10]
print(ab_day_10)
print(ab_day_10_props)

# to find culmative counts:
ab_final = ab_count.drop(columns=['day', 'date']).sum()   # you can technically keep these, but the data is meaningless for those columns
print("abortion totals:")
print(ab_final)

# to find final proportions:
ab_total = ab_final.sum()
ab_final_prop = ab_final / ab_total
print("final abortion proportions:")
print(ab_final_prop)