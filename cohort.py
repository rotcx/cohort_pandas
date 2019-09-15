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
    fmt_type =  '.0%' if normalze else '.0f'
    sns.heatmap(pivoted_df, mask=pivoted_df.isnull(), annot=True, fmt='0.0f')
    plt.yticks(rotation='horizontal')
    display(pivoted_df)
