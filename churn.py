import json
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,StratifiedKFold,cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score,average_precision_score,precision_score,recall_score,f1_score,confusion_matrix,ConfusionMatrixDisplay
P=Path(__file__).parent

def generate(n=1800,seed=42):
    r=np.random.default_rng(seed); tenure=r.integers(1,73,n); tickets=r.poisson(1.6,n); fee=r.uniform(20,120,n); days=r.integers(0,61,n); auto=r.integers(0,2,n)
    logit=-1.3-.025*tenure+.36*tickets+.016*(fee-60)+.045*days-.8*auto
    target=r.binomial(1,1/(1+np.exp(-logit)))
    return pd.DataFrame({'tenure_months':tenure,'support_tickets':tickets,'monthly_fee':fee.round(2),'inactive_days':days,'autopay':auto,'churn':target})

def train(data):
    X=data.drop(columns='churn');y=data.churn
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,stratify=y,random_state=42)
    models={'logistic':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)),'random_forest':RandomForestClassifier(n_estimators=100,min_samples_leaf=7,random_state=42,n_jobs=1)}
    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
    scores={k:float(cross_val_score(m,Xtr,ytr,cv=cv,scoring='average_precision').mean()) for k,m in models.items()}
    selected=max(scores,key=scores.get); model=models[selected].fit(Xtr,ytr); baseline=DummyClassifier(strategy='prior').fit(Xtr,ytr)
    def evaluate(m):
        prob=m.predict_proba(Xte)[:,1]; pred=(prob>=.5).astype(int)
        return {'roc_auc':float(roc_auc_score(yte,prob)),'average_precision':float(average_precision_score(yte,prob)),'precision':float(precision_score(yte,pred,zero_division=0)),'recall':float(recall_score(yte,pred,zero_division=0)),'f1':float(f1_score(yte,pred,zero_division=0)),'confusion_matrix':confusion_matrix(yte,pred,labels=[0,1]).tolist()}
    report={'training_rows':len(Xtr),'test_rows':len(Xte),'test_churn_rate':float(yte.mean()),'cv_average_precision':scores,'selected_model':selected,'threshold':.5,'holdout':evaluate(model),'dummy_baseline':evaluate(baseline)}
    return report

def main():
    data=generate();data.to_csv(P/'synthetic_customers.csv',index=False);report=train(data);(P/'results.json').write_text(json.dumps(report,indent=2));
    fig,ax=plt.subplots(figsize=(5.6,4.6));ConfusionMatrixDisplay(np.array(report['holdout']['confusion_matrix']),display_labels=['Retained','Churned']).plot(ax=ax,colorbar=False,cmap='Blues');ax.set_title('Holdout confusion matrix | synthetic data');fig.tight_layout();fig.savefig(P/'confusion_matrix.png',dpi=140);plt.close(fig);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
