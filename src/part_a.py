
from pathlib import Path
import sqlite3, re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

print(ROOT)
RAW, CLEAN, OUT = ROOT/'data/raw', ROOT/'data/clean', ROOT/'outputs'
DB = ROOT/'data/instamart.db'
for folder in (RAW,CLEAN,OUT): folder.mkdir(parents=True,exist_ok=True)

def load_raw():
    """Load source-of-truth exports without changing the source files."""
    customers=pd.read_csv(RAW/'customers.csv')
    products=pd.read_csv(RAW/'products.csv')
    orders=pd.read_csv(RAW/'orders.csv')
    tickets=pd.read_csv(RAW/'tickets.csv',keep_default_na=False)
    return customers,products,orders,tickets

def audit(customers,products,orders,tickets):
    """Shapes, dtypes, per-column missingness and required raw anomaly counts."""
    for name,df in [('customers',customers),('products',products),('orders',orders),('tickets',tickets)]:
        print('\n',name,'shape:',df.shape)
        print('Types:\n',df.dtypes.to_string())
        missing=df.isna().sum()
        for c in df.select_dtypes('object'):
            missing[c]=df[c].isna().sum()+df[c].fillna('').astype(str).str.strip().eq('').sum()
        print(pd.DataFrame({'missing':missing,'pct':(100*missing/len(df)).round(2)}).to_string())
    age=pd.to_numeric(customers.age,errors='coerce')
    delivery=pd.to_numeric(orders.delivery_minutes,errors='coerce')
    figures={
        'raw order rows':len(orders),
        'exact duplicate rows':int(orders.duplicated().sum()),
        'pending rows':int(orders.status.astype(str).str.strip().str.lower().eq('pending').sum()),
        'distinct order IDs':orders.order_id.nunique(),
        'non-numeric amounts':int(pd.to_numeric(orders.order_amount,errors='coerce').isna().sum()),
        'negative delivery times':int(delivery.lt(0).sum()),
        'blank delivery times':int(orders.delivery_minutes.isna().sum()+orders.delivery_minutes.fillna('').astype(str).str.strip().eq('').sum()-orders.delivery_minutes.isna().sum()),
        'blank ages':int(age.isna().sum()),
        'ages outside 10–100':int((age.notna() & ~age.between(10,100)).sum()),
        'raw city spellings':customers.city.nunique(),
        'nonblank gender spellings':customers.gender.dropna().loc[lambda x:x.astype(str).str.strip().ne('')].nunique(),
        'blank genders':int(customers.gender.fillna('').astype(str).str.strip().eq('').sum())
    }
    result=pd.DataFrame(figures.items(),columns=['audit measure','value'])
    print('\n',result.to_string(index=False))
    return result

def clean_customers(df):
    """Normalize city and gender, impute invalid ages by within-city median."""
    c=df.copy()
    c['city']=c.city.astype('string').str.strip().str.title()
    g=c.gender.fillna('').astype(str).str.strip().str.lower()
    c['gender']=g.map({'m':'male','male':'male','f':'female','female':'female'}).fillna('unknown')
    age=pd.to_numeric(c.age,errors='coerce')
    c['age']=age.where(age.between(10,100))
    c['age']=c.age.fillna(c.groupby('city').age.transform('median'))
    assert c.age.notna().all(), 'City median failed to fill ages'
    c['signup_date']=pd.to_datetime(c.signup_date,errors='raise')
    print('Cities:\n',c.city.value_counts().to_string(),'\nGenders:\n',c.gender.value_counts().to_string())
    return c

def clean_orders(df,products):
    """Apply the specified rules in order, retaining the latest status per ID."""
    o=df.copy()
    o['order_ts']=pd.to_datetime(o.order_ts,errors='raise')
    before=len(o); o=o.drop_duplicates()
    print('Exact duplicates removed:',before-len(o))
    before=len(o)
    o=o.sort_values('order_ts',kind='stable').drop_duplicates('order_id',keep='last').copy()
    print('Older status records removed:',before-len(o))
    o['status']=o.status.astype(str).str.strip().str.lower()
    o['order_amount']=pd.to_numeric(o.order_amount,errors='coerce')
    o['quantity']=pd.to_numeric(o.quantity,errors='raise')
    o['discount_pct']=pd.to_numeric(o.discount_pct,errors='raise')
    prices=products[['product_id','price_inr','category']].copy()
    prices['price_inr']=pd.to_numeric(prices.price_inr,errors='raise')
    o=o.merge(prices,on='product_id',how='left',validate='many_to_one')
    assert o.price_inr.notna().all(),'Unmatched product IDs'
    missing=o.order_amount.isna()
    o.loc[missing,'order_amount']=(o.loc[missing,'price_inr']*o.loc[missing,'quantity']*(1-o.loc[missing,'discount_pct']/100)).round(2)
    print('Amounts reconstructed:',int(missing.sum()))
    o['delivery_minutes']=pd.to_numeric(o.delivery_minutes,errors='coerce')
    negative=o.delivery_minutes.lt(0)
    o.loc[negative,'delivery_minutes']=np.nan
    print('Negative delivery values cleared:',int(negative.sum()))
    completed=o.status.eq('completed')
    med=o.loc[completed].groupby('category').delivery_minutes.transform('median')
    fill=completed & o.delivery_minutes.isna()
    o.loc[completed,'delivery_minutes']=o.loc[completed,'delivery_minutes'].fillna(med)
    print('Completed delivery values imputed:',int(fill.sum()))
    assert o.loc[completed,'delivery_minutes'].notna().all()
    o=o.drop(columns=['price_inr','category'])
    print('Final status counts:\n',o.status.value_counts().to_string())
    return o

def store(customers,products,orders):
    """Write three clean CSVs and SQLite tables; dates stored as ISO-like text."""
    tables={'customers':customers.copy(),'products':products.copy(),'orders':orders.copy()}
    tables['customers']['signup_date']=pd.to_datetime(tables['customers'].signup_date).dt.strftime('%Y-%m-%d')
    tables['orders']['order_ts']=pd.to_datetime(tables['orders'].order_ts).dt.strftime('%Y-%m-%d %H:%M:%S')
    with sqlite3.connect(DB) as conn:
        for name,df in tables.items():
            df.to_csv(CLEAN/f'{name}.csv',index=False)
            df.to_sql(name,conn,if_exists='replace',index=False)
    print('Saved clean CSVs and',DB)
    return tables

def validate(customers,products,orders):
    """Reference assertions; only print the success message when all pass."""
    assert len(customers)==1200
    assert customers.city.nunique()==6
    assert customers.city.value_counts().to_dict()==dict(Pune=227,Delhi=210,Bengaluru=197,Chennai=191,Hyderabad=191,Mumbai=184)
    assert customers.gender.value_counts().to_dict()==dict(female=610,male=560,unknown=30)
    assert customers.age.notna().all() and customers.age.between(18,60).all()
    assert len(orders)==12309 and orders.order_id.is_unique
    assert orders.status.value_counts().to_dict()==dict(completed=11110,cancelled=1199)
    assert orders.order_amount.notna().all()
    assert orders.loc[orders.status.eq('completed'),'delivery_minutes'].notna().all()
    assert orders.loc[orders.status.eq('cancelled'),'delivery_minutes'].isna().sum()==1199
    assert abs(orders.loc[orders.status.eq('completed'),'order_amount'].sum()-2998668.80)<0.011
    print('ALL CHECKPOINTS PASSED')

def run_sql():
    """Run all seven named SQL queries and display their complete results."""
    sql=(ROOT/'sql/queries.sql').read_text()
    chunks=re.split(r'(?m)^-- (Q[1-7])\b[^\n]*\n',sql)
    results={}
    with sqlite3.connect(DB) as conn:
        for i in range(1,len(chunks),2):
            name=chunks[i]; query=chunks[i+1].strip().rstrip(';')
            results[name]=pd.read_sql_query(query,conn)
            print('\n'+name,'rows:',len(results[name]))
            print(results[name].to_string(index=False))
    return results

def pandas_eda(customers,products,orders,sql_results):
    """Cross-check Q2 and calculate city aggregations, distribution and correlations."""
    completed=orders.loc[orders.status.eq('completed')].copy()
    joined=completed.merge(customers[['customer_id','city']],on='customer_id').merge(products[['product_id','category']],on='product_id')
    pivot=pd.pivot_table(joined,index='city',columns='category',values='order_amount',aggfunc='sum',margins=True,fill_value=0)
    print('Pandas pivot:\n',pivot.to_string())
    sql_total=sql_results['Q2'].revenue.sum()
    assert np.isclose(pivot.loc['All','All'],sql_total)
    city=joined.groupby('city').agg(sum=('order_amount','sum'),mean=('order_amount','mean'),count=('order_amount','size'))
    print('\nCity summary:\n',city.to_string())
    x=completed.order_amount
    q1,q2,q3=x.quantile([.25,.5,.75])
    lo,hi=q1-1.5*(q3-q1),q3+1.5*(q3-q1)
    summary=pd.Series({'mean':x.mean(),'median':q2,'skewness':x.skew(),'Q1':q1,'Q3':q3,'lower fence':lo,'upper fence':hi,'below lower':int(x.lt(lo).sum()),'above upper':int(x.gt(hi).sum())})
    print('\nDistribution:\n',summary.to_string())
    corr=completed[['order_amount','discount_pct','quantity','delivery_minutes']].corr()
    print('\nCorrelations:\n',corr.to_string())
    return pivot,city,summary,corr
