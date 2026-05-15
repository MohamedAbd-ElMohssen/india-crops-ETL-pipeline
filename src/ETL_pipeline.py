import pandas as pd
import numpy as np

# !  Load and Filter Raw Dataset
raw_data= pd.read_parquet('data/raw/india_2010_2023.parquet')
selected_crops=raw_data[raw_data['Commodity'].isin(['Apple','Banana','Grapes','Guava','Karbuja (Musk Melon)'
,'Mango','Lemon','Orange','Peach','Water Melon','Pear (Marasebu)','Pomegranate', 'Cowpea (Veg)','Bhindi (Ladies Finger)','Brinjal','Cabbage','Capsicum','Green Chilli',
'Carrot','Cucumbar (Kheera)','Onion','Potato','Tomato','Garlic','Onion Green', 'Rice','Wheat','Maize','Barley (Jau)', 'Ginger (Green)','Black pepper','Cummin Seed (Jeera)','Turmeric','Mint (Pudina)','Coriander (Leaves)'])]

# !  Data exploration
print(raw_data.head(),'\n')
print(raw_data.duplicated().sum())
print(((raw_data['Min_Price'] == 0) | (raw_data['Max_Price'] == 0)).sum())

print('-------------------------------------------------------------------------------------------')

print(selected_crops.count())
print(((selected_crops['Min_Price'] == 0) | (selected_crops['Max_Price'] == 0)).sum())
zero_ratios = (selected_crops[['Min_Price','Max_Price','Modal_Price']] == 0).mean()
print(zero_ratios)
# !  Data cleaning 
selected_crops.loc[:,['Min_Price','Max_Price']] = selected_crops.loc[:,['Min_Price','Max_Price']].replace(0, np.nan)

data_clean=selected_crops.dropna(subset=['Min_Price','Max_Price']).copy()

data_clean.info()
print(((data_clean['Min_Price'] == 0) | (data_clean['Max_Price'] == 0)).sum())
print('-------------------------------------------------------------------------------------------')
print(data_clean.duplicated().sum())
print(data_clean.count())
# !  Renaming Columns
data_clean.rename(columns={
    'Arrival_Date': 'date',
    'Commodity': 'crop_name',
    'Min_Price': 'min_price',
    'Max_Price': 'max_price',
    'Modal_Price': 'avg_price'
}, inplace=True)

# print(data_clean.info())
# ? Rename Crop names

print(data_clean['crop_name'].unique())

crop_names= {
    'Karbuja (Musk Melon)' : 'Cantaloupe',
    'Pear (Marasebu)' : 'Pear',
    'Water Melon' : 'Watermelon',
    'Cowpea (Veg)' : 'Green Beans',
    'Bhindi (Ladies Finger)' : 'Okra',
    'Brinjal' : 'Eggplant',
    'Capsicum' : 'Bell Pepper',
    'Green Chilli' : 'Green Hot Pepper',
    'Cucumbar (Kheera)' : 'Cucumber',
    'Onion Green' : 'Green Onion',
    'Maize' : 'Corn',
    'Barley (Jau)' : 'Barley',
    'Ginger (Green)' : 'Ginger',
    'Cummin Seed (Jeera)' : 'Cumin',
    'Mint (Pudina)' : 'Mint',
    'Coriander (Leaves)' : 'Coriander Leaves'
}

data_clean['crop_name']= data_clean['crop_name'].replace(crop_names)

print(data_clean['crop_name'].unique())

# ! Features(Fields) selection

selected_features=data_clean[['date', 'crop_name', 'min_price', 'max_price', 'avg_price']]

# ! Data Aggregation on All cleaned Data
aggregated_data = selected_features.groupby(['date', 'crop_name']).agg({
    'min_price': 'min',
    'max_price': 'max',
    'avg_price': 'mean'
}).reset_index()

print(aggregated_data.count())
print('-------------------------------------------------------------------------------------------')
aggregated_data.info()

print((aggregated_data['min_price'] <= aggregated_data['avg_price']).all())
print((aggregated_data['avg_price'] <= aggregated_data['max_price']).all())

# ? Add Crop Categories

category_names= {
    'Apple' : 'Fruits',
    'Banana' : 'Fruits',
    'Grapes' : 'Fruits',
    'Guava' : 'Fruits',
    'Cantaloupe' : 'Fruits',
    'Mango' : 'Fruits',
    'Lemon' : 'Fruits',
    'Orange' : 'Fruits',
    'Peach' : 'Fruits',
    'Pear' : 'Fruits',
    'Pomegranate' : 'Fruits',
    'Watermelon' : 'Fruits',
    'Green Beans' : 'Vegetables',
    'Okra' : 'Vegetables',
    'Eggplant' : 'Vegetables',
    'Cabbage' : 'Vegetables',
    'Bell Pepper' : 'Vegetables',
    'Green Hot Pepper' : 'Vegetables',
    'Carrot' : 'Vegetables',
    'Cucumber' : 'Vegetables',
    'Onion' : 'Vegetables',
    'Potato' : 'Vegetables',
    'Tomato' : 'Vegetables',
    'Garlic' : 'Vegetables',
    'Green Onion' : 'Vegetables',
    'Rice' : 'Grains',
    'Wheat' : 'Grains',
    'Corn' : 'Grains',
    'Barley' : 'Grains',
    'Ginger' : 'Herbs & Spices',
    'Black pepper' : 'Herbs & Spices',
    'Cumin' : 'Herbs & Spices',
    'Turmeric' : 'Herbs & Spices',
    'Mint' : 'Herbs & Spices',
    'Coriander Leaves' : 'Herbs & Spices'
}

aggregated_data['category'] = aggregated_data['crop_name'].map(category_names)
aggregated_data= aggregated_data[['date', 'crop_name', 'category', 'min_price', 'max_price', 'avg_price']]
aggregated_data.info()
print(aggregated_data[aggregated_data['category'].isna()])
print(aggregated_data.head())

# ? Convert Prices from Quintal(prices per 100 kilogram) to Kilogram

data_crops_india_per_quintal=aggregated_data.sort_values('date', ascending=True)
data_crops_india_per_quintal['avg_price'] = data_crops_india_per_quintal['avg_price'].round(2)
data_crops_india_per_quintal[['min_price','max_price','avg_price']] = data_crops_india_per_quintal[['min_price','max_price','avg_price']] / 100


# ? Handle Price Outliers
ratio = 0.2

rule = (data_crops_india_per_quintal['min_price'] < 2) | (data_crops_india_per_quintal['max_price'] > 10000)

data_crops_india_per_quintal.loc[rule, 'min_price'] = data_crops_india_per_quintal.loc[rule, 'avg_price'] * (1 - ratio)
data_crops_india_per_quintal.loc[rule, 'max_price'] = data_crops_india_per_quintal.loc[rule, 'avg_price'] * (1 + ratio)

data_crops_india_per_quintal[['min_price','max_price','avg_price']] = data_crops_india_per_quintal[['min_price','max_price','avg_price']].round(2)

data_crops_india_per_quintal['date'] = pd.to_datetime(data_crops_india_per_quintal['date'])
data_crops_india_per_quintal = data_crops_india_per_quintal[(data_crops_india_per_quintal['avg_price'] > 0)]
# ? Data Validation
print((data_crops_india_per_quintal[['min_price','max_price','avg_price']] <= 0).sum())
print((data_crops_india_per_quintal[['min_price','max_price','avg_price']] == 0).sum())
print(data_crops_india_per_quintal.duplicated().sum())
print((data_crops_india_per_quintal['min_price'] <= data_crops_india_per_quintal['avg_price']).all())
print((data_crops_india_per_quintal['avg_price'] <= data_crops_india_per_quintal['max_price']).all())
print(data_crops_india_per_quintal.groupby(['date','crop_name']).size().max())
print(data_crops_india_per_quintal.dtypes)
print(data_crops_india_per_quintal.head(10))
data_crops_india_per_quintal.info()
print(data_crops_india_per_quintal[(data_crops_india_per_quintal['min_price'] > data_crops_india_per_quintal['max_price'])].sum())

# ! Export Processed Dataset

data_crops_india_per_quintal.to_csv('data/processed/data_crops_india_per_kilo.csv', index=False)

data_2023 = data_crops_india_per_quintal[data_crops_india_per_quintal['date'].dt.year == 2023]
data_2023.to_csv('data/processed/data_crops_india_per_kilo_2023.csv', index=False)