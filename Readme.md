



# Cohort Chart with pandas



Cohort chart simply shows user retention tendency and is widely used for user churn management.

Here, I will introduce how to make cohort chart using pandas.



### Data load

we will use Kaggle retail transaction data as an example

you can download here :

https://www.kaggle.com/puneetbhaya/online-retail



Read file and drop rows with a NaN value in the CustomerID.

```python
df=pd.read_excel('Online Retail.xlsx')
df=df.dropna(subset=['CustomerID'])
df.head()
```

![1568529247081](Cohort%20Chart%20with%20pandas.assets/1568529247081.png)



### define some variables

Let's define each column name

```python
date_column='InvoiceDate'
id_column='CustomerID'
start_date='2012-01-01'
freq = 'MS' ## cohort gap period
```

(here MS frequency means monthly and s



Next, convert date related column to datetime column

```python
df[date_column]=pd.to_datetime(df[date_column]).dt.floor('d')
```



## Preprocessing for Cohort chart

We have to make Cohort Group for figuring out when users joined our service

```python
df['CohortGroup']=df.groupby(['sid_id'])[date_column].transform(min)
```

![1568529549841](Cohort%20Chart%20with%20pandas.assets/1568529549841.png)



After this, we have to groupby CohortGroup and InvoceDate column, and get the number of unique customer IDs. We can apply pd.Grouper function which is very useful for grouping by specific date period. 

```python
df_grouped=df.groupby([pd.Grouper(key='CohortGroup',freq=freq, closed='left', label='left'),pd.Grouper(key=date_column,freq=freq, closed='left', label='left')])
df=df_grouped[id_column].nunique().rename('id_nuniq').to_frame()
```

![1568531071377](Cohort%20Chart%20with%20pandas.assets/1568531071377.png)



Here, we have to **be careful** in case there is no cohort period that user singed up (But it seems this data is not that case)

So, have have to insert empty Cohort Period by resampling

```python
df.reset_index(inplace=True)
df=df.groupby('CohortGroup').apply(lambda row : row.resample(freq,on=date_column).sum().fillna(0))
```



and then, assign Cohort period to each InvoiceDate Period

```python
df=df.groupby('CohortGroup').apply(cohort_period)
df.reset_index(inplace=True)
```

![1568531457027](Cohort%20Chart%20with%20pandas.assets/1568531457027.png)



Last part we have to **be aware of** is that CohortGroup could have also empty value (but this data is also not). So we have to deal with it.

```python
df.reset_index(inplace=True)    
base_df=pd.DataFrame(index = pd.date_range(df['CohortGroup'].min(), df['CohortGroup'].max(), freq=freq, closed=None))
df=pd.merge(base_df,df, how='left', left_index=True, right_on='CohortGroup')      
```



Finally, we pivot this df, and convert index datetime type to str for plot.

```python
df.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)      
df=df.reset_index().dropna()
final=pd.pivot_table(df, index='CohortGroup', columns='CohortPeriod', values='id_nuniq')
final.index=final.index.strftime('%Y-%m-%d')  
```

![1568531809326](Cohort%20Chart%20with%20pandas.assets/1568531809326.png)

We can quickly figure out how many users keep transaction under each cohort group 



If  you want to see user **retention rate** rather then the number of users, we can normalize

```python
final=final.divide(final[1],axis=0)
```



![1568532201311](Cohort%20Chart%20with%20pandas.assets/1568532201311.png)



## Different cohort period

We can make cohort chart not only monthly but also 3monthly, weekly, 2weekly, and w-MON, TUE..(start day defiend) etc with Pandas pd.Grouper()

![1568533711722](Cohort%20Chart%20with%20pandas.assets/1568533711722.png)

![1568533721835](Cohort%20Chart%20with%20pandas.assets/1568533721835.png)

To see more frequency options : (https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html)



For example, if we set freq to W-MON, cohort group value is set to be Monday.

![1568547535874](Cohort%20Chart%20with%20pandas.assets/1568547535874.png)

...

![1568547546592](Cohort%20Chart%20with%20pandas.assets/1568547546592.png)



As you can see, cohort group is changed on Monday basis.





## sns seabon plot

```python
def sns_cohort_plot(pivoted_df, normalize=False):            
    plt.figure(figsize=(10,10))        
    plt.title('Cohort retention')    
    fmt_type =  '.0%' if normalze else '.0f'    
    sns.heatmap(pivoted_df, mask=pivoted_df.isnull(), annot=True, fmt='0.0f')
    plt.yticks(rotation='horizontal')
    display(pivoted_df)
```

```python
sns_cohort(final)
```







![1568553165190](Cohort%20Chart%20with%20pandas.assets/1568553165190.png)



## All together

```python
%matplotlib inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def cohort(input_df, date_column, id_column, start_date='1970-01-01', freq='MS', normalize=False):
    def cohort_period(df):
        df['CohortPeriod'] = np.arange(len(df)) + 1
        return df

    df = input_df.copy()

    df['CohortGroup'] = df.groupby([id_column])[date_column].transform(min)
    df = df[df['CohortGroup'] >= start_date]

    df_grouped = df.groupby([pd.Grouper(key='CohortGroup', freq=freq, closed='left', label='left'),
                             pd.Grouper(key=date_column, freq=freq, closed='left', label='left')])
    df = df_grouped[id_column].nunique().rename('id_nuniq').to_frame()

    df.reset_index(inplace=True)
    df = df.groupby('CohortGroup').apply(lambda row: row.resample(freq, on=date_column).sum().fillna(0))
    df = df.groupby('CohortGroup').apply(cohort_period)

    df.reset_index(inplace=True)
    base_df = pd.DataFrame(
        index=pd.date_range(df['CohortGroup'].min(), df['CohortGroup'].max(), freq=freq, closed=None))
    df = pd.merge(base_df, df, how='left', left_index=True, right_on='CohortGroup')

    df.set_index(['CohortGroup', 'CohortPeriod'], inplace=True)
    df = df.reset_index().dropna()
    final = pd.pivot_table(df, index='CohortGroup', columns='CohortPeriod', values='id_nuniq')
    final.index = final.index.strftime('%Y-%m-%d')

    if normalize:
        final = final.divide(final[1], axis=0)

    return final

def sns_cohort_plot(pivoted_df, normalize=False):            
    plt.figure(figsize=(10,10))        
    plt.title('Cohort retention')    
    fmt_type =  '.0%' if normalize else '.0f'    
    sns.heatmap(pivoted_df, mask=pivoted_df.isnull(), annot=True, fmt='0.0f')
    plt.yticks(rotation='horizontal')
    display(pivoted_df)
```



