#Detecting every NaN value on every columns
df_ecommerce.isnull().sum()

#Removing data with NaN value
prev_size = df_ecommerce.shape[0]
df_ecommerce.dropna(how='any',inplace=True)
current_size = df_ecommerce.shape[0]

df_ecommerce.isnull().values.any()
print(df_ecommerce.shape)
print('From the data cleaning, we remove {}% of NaN value data'.format(round(((prev_size - current_size)/prev_size)*100,2)))

#Converting the timestamp format data to date data
df_ecommerce['order_purchase_timestamp'] = pd.to_datetime(df_ecommerce['order_purchase_timestamp']).dt.date
df_ecommerce['order_estimated_delivery_date'] = pd.to_datetime(df_ecommerce['order_estimated_delivery_date']).dt.date
df_ecommerce['order_delivered_customer_date'] = pd.to_datetime(df_ecommerce['order_delivered_customer_date']).dt.date
df_ecommerce['shipping_limit_date'] = pd.to_datetime(df_ecommerce['shipping_limit_date']).dt.date

#Converting date time into string to remove the timestamp notation
df_ecommerce['delivery_days'] = df_ecommerce['order_delivered_customer_date'].sub(df_ecommerce['order_purchase_timestamp'],axis=0).astype(str)
df_ecommerce['estimated_days'] = df_ecommerce['order_estimated_delivery_date'].sub(df_ecommerce['order_purchase_timestamp'],axis=0).astype(str)
df_ecommerce['shipping_days'] = df_ecommerce['shipping_limit_date'].sub(df_ecommerce['order_purchase_timestamp'],axis=0).astype(str)

#Replacing the time stamp notation and converting type to int
df_ecommerce['delivery_days'] = df_ecommerce['delivery_days'].str.replace(" days","").astype(int)
df_ecommerce['estimated_days'] = df_ecommerce['estimated_days'].str.replace(" days","").astype(int)
df_ecommerce['shipping_days'] = df_ecommerce['shipping_days'].str.replace(" days","").astype(int)
df_ecommerce.drop(['order_purchase_timestamp', 'order_delivered_customer_date', 'order_estimated_delivery_date',
                   'shipping_limit_date'],axis=1,inplace=True)
                   
#Result of data preprocessing
df_ecommerce.head()
