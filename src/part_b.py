from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, accuracy_score, precision_recall_fscore_support, confusion_matrix, f1_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
print(ROOT)
DATA = ROOT / 'data' / 'clean'
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)
SNAPSHOT = pd.Timestamp('2025-10-01')
CUTOFF = pd.Timestamp('2025-09-01')
NUMERIC = ['recency','frequency','monetary','avg_discount_pct','cancellation_rate','avg_delivery_minutes','age','tenure_days']
CATEGORICAL = ['city','gender','favourite_category']


def load_data():
    orders = pd.read_csv(DATA / 'orders.csv')
    customers = pd.read_csv(DATA / 'customers.csv')
    products = pd.read_csv(DATA / 'products.csv')
    required_o = [
        'customer_id', 'order_id', 'product_id',
        'order_ts', 'status', 'order_amount',
        'discount_pct', 'delivery_minutes'
    ]
    required_c = [
        'customer_id', 'city', 'age', 'gender', 'signup_date'
    ]
    required_p = ['product_id', 'category']
    for label, frame, cols in [
        ('orders', orders, required_o),
        ('customers', customers, required_c),
        ('products', products, required_p)
    ]:
        missing = set(cols) - set(frame.columns)
        if missing:
            raise ValueError(
                f'{label} missing columns: {sorted(missing)}. '
                'Adjust mapping to Part A schema.')
    orders['order_ts'] = pd.to_datetime(orders['order_ts'])
    customers['signup_date'] = pd.to_datetime(
        customers['signup_date']
    )
    orders['status'] = (
        orders['status'].astype(str).str.strip().str.lower()
    )
    for col in ['order_amount', 'discount_pct', 'delivery_minutes']:
        orders[col] = pd.to_numeric(orders[col], errors='coerce')

    customers['age'] = pd.to_numeric(
        customers['age'], errors='coerce'
    )
    orders = orders.merge(
        products[['product_id', 'category']],
        on='product_id',
        how='left',
        validate='many_to_one'
    )

    if orders['category'].isna().any():
        print(
            'Warning: Some orders do not have a matching product category.'
        )

    return orders, customers


def build_rfm(orders):
    completed = orders[(orders.status=='completed') & (orders.order_ts < SNAPSHOT)]
    rfm = completed.groupby('customer_id').agg(last_order_ts=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum')).reset_index()
    rfm.insert(1,'recency',(SNAPSHOT-rfm.pop('last_order_ts')).dt.days)
    print('RFM customers:',len(rfm),'median/min/max recency:',rfm.recency.median(),rfm.recency.min(),rfm.recency.max())
    print('Frequency mean/max:',rfm.frequency.mean(),rfm.frequency.max(),'Monetary mean/max:',rfm.monetary.mean(),rfm.monetary.max())
    return rfm

def cluster_segments(rfm, k=4):
    scaled = StandardScaler().fit_transform(rfm[['recency','frequency','monetary']])
    rows=[]
    for n in range(2,9):
        model=KMeans(n_clusters=n,n_init=10,random_state=42).fit(scaled)
        rows.append({'K':n,'inertia':model.inertia_,'silhouette':silhouette_score(scaled,model.labels_)})
    scores=pd.DataFrame(rows)
    print(scores.to_string(index=False))
    print('Interpretation: K=4 is the reference choice: strong silhouette and an actionable elbow. Inertia always falls with more clusters because extra centroids cannot worsen the optimum; it does not penalize complexity or measure usefulness.')
    model=KMeans(n_clusters=k,n_init=10,random_state=42).fit(scaled)
    result=rfm.copy(); result['segment_id']=model.labels_
    profile=result.groupby('segment_id').agg(size=('customer_id','size'),recency=('recency','mean'),frequency=('frequency','mean'),monetary=('monetary','mean'))
    premium=profile.monetary.idxmax()
    frequent=profile.drop(index=premium).frequency.idxmax()
    remaining=[i for i in profile.index if i not in (premium,frequent)]
    names={premium:'Premium',frequent:'Frequent Loyal'}
    for i in remaining:
        names[i]='At Risk' if profile.loc[i,'recency']==profile.loc[remaining,'recency'].max() else 'Occasional'
    actions={'Premium':'VIP care and personalized rewards','Frequent Loyal':'Loyalty perks and cross-sell','At Risk':'Win-back campaign','Occasional':'Frequency-building offers'}
    profile['segment_name']=[names[i] for i in profile.index]
    profile['recommended_action']=[actions[names[i]] for i in profile.index]
    result['segment_name']=result.segment_id.map(names)
    result['is_premium']=result.segment_id.eq(premium)
    pca=PCA(n_components=2).fit(scaled)
    print('\nSegment profile (cluster IDs are arbitrary labels, not ranks):\n',profile.to_string())
    print('PCA explained variance:',pca.explained_variance_ratio_,'total:',sum(pca.explained_variance_ratio_))
    result.to_csv(OUT/'customer_segments.csv',index=False)
    return result,scores,profile,pca

def build_lapse_cohort(orders, customers):
    past=orders[orders.order_ts < CUTOFF].copy()
    completed=past[past.status=='completed'].copy()
    base=completed.groupby('customer_id').agg(last_order=('order_ts','max'),frequency=('order_id','nunique'),monetary=('order_amount','sum'),avg_discount_pct=('discount_pct','mean'),avg_delivery_minutes=('delivery_minutes','mean')).reset_index()
    base['recency']=(CUTOFF-base.pop('last_order')).dt.days
    final=past[past.status.isin(['completed','cancelled'])]
    cancel=final.groupby('customer_id').status.agg(lambda s:(s=='cancelled').mean()).rename('cancellation_rate')
    base=base.merge(cancel,on='customer_id',how='left')
    category=(completed.groupby(['customer_id','category']).size().reset_index(name='n').sort_values(['customer_id','n','category'],ascending=[True,False,True]).drop_duplicates('customer_id')[['customer_id','category']].rename(columns={'category':'favourite_category'}))
    base=base.merge(category,on='customer_id',how='left').merge(customers[['customer_id','city','age','gender','signup_date']],on='customer_id',how='left',validate='one_to_one')
    base['tenure_days']=(CUTOFF-base.pop('signup_date')).dt.days
    sept=orders[(orders.status=='completed') & (orders.order_ts>=CUTOFF) & (orders.order_ts<SNAPSHOT)].customer_id.unique()
    base['lapsed']=(~base.customer_id.isin(sept)).astype(int)
    features=NUMERIC+CATEGORICAL
    if base[features].isna().any().any():
        raise ValueError('Missing features; clean Part A data first:\n'+str(base[features].isna().sum()[lambda x:x>0]))
    assert not base[features].isna().any().any()
    print('Cohort:',len(base),'lapsed:',base.lapsed.sum(),f'({base.lapsed.mean():.2%})')
    return base,features

def make_pipeline(estimator):
    pre=ColumnTransformer([('num',StandardScaler(),NUMERIC),('cat',OneHotEncoder(drop='first',handle_unknown='ignore'),CATEGORICAL)])
    return Pipeline([('preprocess',pre),('model',estimator)])

def metrics(model,X,y,label):
    pred=model.predict(X)
    p,r,f,_=precision_recall_fscore_support(y,pred,average='binary',pos_label=1,zero_division=0)
    print(f'{label}: accuracy={accuracy_score(y,pred):.3f} precision={p:.3f} recall={r:.3f} F1={f:.3f}\nConfusion matrix [0,1]:\n{confusion_matrix(y,pred,labels=[0,1])}')
    return {'precision':p,'recall':r,'f1':f,'predictions':pred}

def train_evaluate(cohort,features):
    X=cohort[features]; y=cohort.lapsed
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
    print('Train/test:',len(Xtr),len(Xte),'Class counts train:',ytr.value_counts().to_dict())
    baseline=np.zeros(len(yte),dtype=int)
    p,r,f,_=precision_recall_fscore_support(yte,baseline,average='binary',zero_division=0)
    print(f'Baseline accuracy={accuracy_score(yte,baseline):.3f}, lapsed precision={p:.3f}, recall={r:.3f}, F1={f:.3f}')
    lr=GridSearchCV(make_pipeline(LogisticRegression(max_iter=1000,random_state=42)),{'model__C':[.01,.1,1,10,100]},cv=5,scoring='f1',n_jobs=-1)
    rf=GridSearchCV(make_pipeline(RandomForestClassifier(n_estimators=100,random_state=42)),{'model__max_depth':[3,5,8],'model__min_samples_leaf':[5,10,20]},cv=5,scoring='f1',n_jobs=-1)
    lr.fit(Xtr,ytr); rf.fit(Xtr,ytr)
    print('Logistic best:',lr.best_params_,'CV F1:',lr.best_score_)
    print('Forest best:',rf.best_params_,'CV F1:',rf.best_score_)
    lm=metrics(lr.best_estimator_,Xte,yte,'Logistic')
    fm=metrics(rf.best_estimator_,Xte,yte,'Forest')
    scaler=lr.best_estimator_.named_steps['preprocess'].named_transformers_['num']
    print('Training frequency scaler mean:',scaler.mean_[NUMERIC.index('frequency')],'Full cohort mean:',X.frequency.mean(),'They differ because the scaler sees only the randomly selected training rows.')
    return {'Xtr':Xtr,'Xte':Xte,'ytr':ytr,'yte':yte,'lr':lr,'rf':rf,'lr_metrics':lm,'rf_metrics':fm,'cohort':cohort}

def balance_experiment(run):
    Xtr,ytr=run['Xtr'],run['ytr']
    counts=ytr.value_counts(); majority=counts.idxmax(); minority=1
    if counts[minority]>=counts[majority]: raise ValueError('Expected lapsed class to be minority')
    rng=np.random.RandomState(42)
    extra=rng.choice(ytr.index[ytr==minority].to_numpy(),size=counts[majority]-counts[minority],replace=True)
    Xbal=pd.concat([Xtr,Xtr.loc[extra]],ignore_index=True)
    ybal=pd.concat([ytr,ytr.loc[extra]],ignore_index=True)
    model=make_pipeline(LogisticRegression(max_iter=1000,C=run['lr'].best_params_['model__C'],random_state=42))
    model.fit(Xbal,ybal)
    print('Balanced train counts:',ybal.value_counts().to_dict())
    print('Original logistic:',{k:run['lr_metrics'][k] for k in ('precision','recall','f1')})
    balanced=metrics(model,run['Xte'],run['yte'],'Oversampled logistic')
    print('Metric changes:',{k:round(balanced[k]-run['lr_metrics'][k],3) for k in ('precision','recall','f1')})
    return model,balanced

def interpret_persist_retain(run,segments,selected='forest'):
    forest=run['rf'].best_estimator_; logistic=run['lr'].best_estimator_
    rf_names=forest.named_steps['preprocess'].get_feature_names_out()
    imp=pd.Series(forest.named_steps['model'].feature_importances_,index=rf_names).sort_values(ascending=False)
    print('Top five forest features:\n',imp.head(5).to_string())
    coefs=pd.Series(logistic.named_steps['model'].coef_[0],index=logistic.named_steps['preprocess'].get_feature_names_out())
    print('Logistic coefficient signs (positive increases lapse odds):\n',coefs.loc[[f'num__{n}' for n in ['recency','frequency','monetary']]].to_string())
    chosen=forest if selected=='forest' else logistic
    score=run['rf_metrics']['f1'] if selected=='forest' else run['lr_metrics']['f1']
    print('Selected:',selected,'test F1:',score)
    if score<.55: print('WARNING: selected model is below acceptance F1; inspect source data/features.')
    joblib.dump(chosen,OUT/'lapse_model.joblib')
    restored=joblib.load(OUT/'lapse_model.joblib')
    assert np.array_equal(chosen.predict(run['Xte']),restored.predict(run['Xte']))
    print('Reloaded predictions exactly match.')
    ranked=pd.DataFrame({'customer_id':run['cohort'].loc[run['Xte'].index,'customer_id'].to_numpy(),'lapse_probability':restored.predict_proba(run['Xte'])[:,1],'actually_lapsed':run['yte'].to_numpy()})
    premium_ids=set(segments.loc[segments.is_premium,'customer_id'])
    ranked=ranked[ranked.customer_id.isin(premium_ids)].sort_values('lapse_probability',ascending=False).head(10)
    ranked.to_csv(OUT/'retention_list.csv',index=False)
    print('Premium test customers shortlisted:',len(ranked),'truly lapsed:',ranked.actually_lapsed.sum())
    print('LIMITATION: Premium is a post-hoc whole-window label including September; this retention list is illustrative, not a leakage-free evaluation.')
    return ranked,imp,coefs
