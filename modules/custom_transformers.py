# Reference
# http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from functools import reduce
from statsmodels.stats.outliers_influence import variance_inflation_factor    
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, mutual_info_regression, RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler, RobustScaler, MultiLabelBinarizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, accuracy_score
from sklearn.tree import DecisionTreeRegressor
from collections import defaultdict
from sklearn.decomposition import PCA
from pdb import set_trace
        

class DF_RemoveMulticolinear(BaseEstimator, TransformerMixin):
    # FunctionTransformer but for pandas DataFrames

    def __init__(self, thresh = 5):
        self.thresh = thresh

    def fit(self, X, y=None):
        # stateless transformer
        cols = X.columns
        variables = np.arange(X.shape[1])
        dropped=True
        while dropped:
            dropped=False
            c = X[cols[variables]].values
            vif = [variance_inflation_factor(c, ix) for ix in np.arange(c.shape[1])]

            maxloc = vif.index(max(vif))
            if max(vif) > self.thresh:
                print('dropping \'' + X[cols[variables]].columns[maxloc] + '\' at index: ' + str(maxloc))
                variables = np.delete(variables, maxloc)
                dropped=True

        print('Remaining variables:')
        print(X.columns[variables])
        self.cols = X.columns[variables]
        return self

    def transform(self, X):
        return X[self.cols]

class DFFunctionTransformer(BaseEstimator, TransformerMixin):
    # FunctionTransformer but for pandas DataFrames

    def __init__(self, *args, **kwargs):
        self.ft = FunctionTransformer(*args, **kwargs)

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        Xt = self.ft.transform(X)
        Xt = pd.DataFrame(Xt, index=X.index, columns=X.columns)
        return Xt


class DFFeatureUnion(BaseEstimator, TransformerMixin):
    # FeatureUnion but for pandas DataFrames

    def __init__(self, transformer_list):
        self.transformer_list = transformer_list

    def fit(self, X, y=None):
        for (name, t) in self.transformer_list:
            t.fit(X, y)
        return self

    def get_feature_names(self):
        return self.columns

    def transform(self, X):
        # assumes X is a DataFrame
        Xts = [t.transform(X) for _, t in self.transformer_list]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xts)
        self.columns = Xunion.columns.tolist()
        return Xunion


class DFImputer(BaseEstimator, TransformerMixin):
    # Imputer but for pandas DataFrames

    def __init__(self, strategy='mean'):
        self.strategy = strategy
        self.imp = None
        self.statistics_ = None

    def fit(self, X, y=None):
        self.imp = Imputer(strategy=self.strategy)
        self.imp.fit(X)
        self.statistics_ = pd.Series(self.imp.statistics_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Ximp = self.imp.transform(X)
        Xfilled = pd.DataFrame(Ximp, index=X.index, columns=X.columns)
        return Xfilled


class DFStandardScaler(BaseEstimator, TransformerMixin):
    # StandardScaler but for pandas DataFrames

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = StandardScaler()
        self.ss.fit(X)
        self.mean_ = pd.Series(self.ss.mean_, index=X.columns)
        self.scale_ = pd.Series(self.ss.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled

class DFMinMaxScaler(BaseEstimator, TransformerMixin):
        # StandardScaler but for pandas DataFrames

    def __init__(self):
        self.ss = None
        self.mean_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.ss = MinMaxScaler()
        self.ss.fit(X)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xss = self.ss.transform(X)
        Xscaled = pd.DataFrame(Xss, index=X.index, columns=X.columns)
        return Xscaled


class DFRobustScaler(BaseEstimator, TransformerMixin):
    # RobustScaler but for pandas DataFrames

    def __init__(self):
        self.rs = None
        self.center_ = None
        self.scale_ = None

    def fit(self, X, y=None):
        self.rs = RobustScaler()
        self.rs.fit(X)
        self.center_ = pd.Series(self.rs.center_, index=X.columns)
        self.scale_ = pd.Series(self.rs.scale_, index=X.columns)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xrs = self.rs.transform(X)
        Xscaled = pd.DataFrame(Xrs, index=X.index, columns=X.columns)
        return Xscaled


class ColumnExtractor(BaseEstimator, TransformerMixin):

    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xcols = X[self.cols]
        return Xcols


class ZeroFillTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xz = X.fillna(value=0)
        return Xz



class Log1pTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xlog = np.log1p(X)
        return Xlog


class DateFormatter(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdate = X.apply(pd.to_datetime)
        return Xdate


class DateDiffer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        beg_cols = X.columns[:-1]
        end_cols = X.columns[1:]
        Xbeg = X[beg_cols].as_matrix()
        Xend = X[end_cols].as_matrix()
        Xd = (Xend - Xbeg) / np.timedelta64(1, 'D')
        diff_cols = ['->'.join(pair) for pair in zip(beg_cols, end_cols)]
        Xdiff = pd.DataFrame(Xd, index=X.index, columns=diff_cols)
        return Xdiff


class DummyTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.dv = None

    def fit(self, X, y=None):
        # assumes all columns of X are strings
        Xdict = X.to_dict('records')
        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(Xdict)
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xdict = X.to_dict('records')
        Xt = self.dv.transform(Xdict)
        cols = self.dv.get_feature_names()
        Xdum = pd.DataFrame(Xt, index=X.index, columns=cols)
        # drop column indicating NaNs
        nan_cols = [c for c in cols if '=' not in c]
        Xdum = Xdum.drop(nan_cols, axis=1)
        return Xdum


class MultiEncoder(BaseEstimator, TransformerMixin):
    # Multiple-column MultiLabelBinarizer for pandas DataFrames

    def __init__(self, sep=','):
        self.sep = sep
        self.mlbs = None

    def _col_transform(self, x, mlb):
        cols = [''.join([x.name, '=', c]) for c in mlb.classes_]
        xmlb = mlb.transform(x)
        xdf = pd.DataFrame(xmlb, index=x.index, columns=cols)
        return xdf

    def fit(self, X, y=None):
        Xsplit = X.applymap(lambda x: x.split(self.sep))
        self.mlbs = [MultiLabelBinarizer().fit(Xsplit[c]) for c in X.columns]
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xsplit = X.applymap(lambda x: x.split(self.sep))
        Xmlbs = [self._col_transform(Xsplit[c], self.mlbs[i])
                 for i, c in enumerate(X.columns)]
        Xunion = reduce(lambda X1, X2: pd.merge(X1, X2, left_index=True, right_index=True), Xmlbs)
        return Xunion


class StringTransformer(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xstr = X.applymap(str)
        return Xstr


class ClipTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, a_min, a_max):
        self.a_min = a_min
        self.a_max = a_max

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xclip = np.clip(X, self.a_min, self.a_max)
        return Xclip


class AddConstantTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, c=1):
        self.c = c

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        Xc = X + self.c
        return Xc


class ShiftTranformer(BaseEstimator, TransformerMixin):
    
    def __init__(self, par = 1):
        self.par = par

    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        # assumes X is a DataFrame
        X_sub = self.par - X 
        return X_sub




class DFRecursiveFeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self,estimator=DecisionTreeRegressor(), n_features = 10, step=10):
        self.estimator = estimator
        self.n_features = n_features
        self.step = step
        
    def fit(self, X, y=None):
        # stateless transformer
        self.RFE = RFE(estimator=self.estimator, n_features_to_select=self.n_features, step=self.step)
        self.RFE.fit(X,y)
        return self

    def transform(self, X):
        return X.loc[:,self.RFE.get_support()]

       


class DFSelectKBest(BaseEstimator, TransformerMixin):
    
    def __init__(self,estimator=mutual_info_regression, n_features = 10):
        self.estimator = estimator
        self.n_features = n_features
        
    def fit(self, X, y=None):
        # stateless transformer
        self.SelectKBest = SelectKBest(self.estimator, k=self.n_features)
        self.SelectKBest.fit(X,y)
        return self

    def transform(self, X):
        return X.loc[:,self.SelectKBest.get_support()]

class DFembedFeatureSelection(BaseEstimator, TransformerMixin):
    
    def __init__(self,estimator=DecisionTreeRegressor(), n_features = 10):
        self.estimator = estimator
        self.n_features = n_features
        
    def fit(self, X, y=None):
        # stateless transformer
        self.estimator.fit(X,y)
        return self

    def transform(self, X):
        ind = self.estimator.feature_importances_.argsort()[-self.n_features:][::-1]
        return X.iloc[:,ind]

class DFadd_ColInteraction(BaseEstimator, TransformerMixin):
    
    def __init__(self,col1,col2):
        self.col1 = col1
        self.col2 = col2
        
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        X[f'{self.col1}_&_{self.col2}'] = X[self.col1]*X[self.col2]
        return X

class DFremoveOutlier_IQR(BaseEstimator, TransformerMixin):
    
    def __init__(self,factor = 1.5):
        self.factor = factor

        
    def fit(self, X, y=None):
        # stateless transformer
        self.y = y
        return self

    def transform(self, X):

        Q1 = X.quantile(0.25)
        Q3 = X.quantile(0.75)
        IQR = Q3 - Q1

        inds = ~((X < (Q1 - self.factor * IQR)) |(X > (Q3 + self.factor * IQR))).any(axis=1)
        X_screened = X[inds]
        self.y = self.y[inds]
        return X_screened, self.y

class DF_RFECV_FeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self,estimator = DecisionTreeRegressor(), cv = StratifiedKFold(3),step = 1, scoring = 'r2'):
        self.estimator = estimator
        self.scoring = scoring
        self.step = step
        self.cv = cv

    def fit(self,X, y=None):

        self.rfevc = RFECV(estimator = self.estimator,step=self.step, cv=self.cv, scoring=self.scoring)
        self.fit(X,y)
        return self

    def transform(self,X):
        return X.drop(X.columns[np.where(self.rfevc.support_ == False)[0]], axis=1, inplace=True)

class DFdrop_Cols(BaseEstimator, TransformerMixin):
    
    def __init__(self,cols = None):
        self.cols = cols
        
    def fit(self, X, y=None):
        # stateless transformer
        return self

    def transform(self, X):
        X.drop(self.cols, axis=1, inplace=True)
        return X

class DF_Model(BaseEstimator, TransformerMixin):
    
    def __init__(self,estimator = None):
        self.estimator = estimator

    def fit(self,X,y=None):
        self.cols = list(X.columns)
        self.estimator.fit(X,y)
        return self

    def predict(self,X):
        preds = self.estimator.predict(X)
        return preds
    
    def get_feature_names(self):
        return self.cols

    def predict_proba(self,X):
        return self.estimator.predict_proba(X)[:,-1]

class DF_PolynomialFeatures(BaseEstimator, TransformerMixin):
    
    def __init__(self,degree = 2, interaction= True):
        self.degree = degree
        self.interaction = interaction
    
    def fit(self,X,y=None):
        self.poly = PolynomialFeatures(degree=self.degree
                                       ,interaction_only=self.interaction)
        self.poly.fit(X)
        return self

    def transform(self,X):
        X_poly = self.poly.transform(X)
        return pd.DataFrame(X_poly,index=X.index,columns=self.poly.get_feature_names(X.columns))

class DF_Tarnsform(BaseEstimator, TransformerMixin):

    def __init__(self,func = None):
        self.func = func

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transfromed = X.apply(self.func)
        return X_transfromed

class DF_PCAtransform(BaseEstimator, TransformerMixin):

    def __init__(self,n_components = 3):
        self.n_components = n_components
    
    def fit(self, X, y=None):
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.fit_transform(X)

class DF_SmoteOverSampler(BaseEstimator, TransformerMixin):
    def __init__(self, ratio = 1):
        self.ratio = ratio
    
    def fit(self, X, y=None):
        self.smote = SMOTE(sampling_strategy = self.ratio, n_jobs=-1).fit(X, y)

    def fit_transform(self, X, y=None):
        self.fit(X, y) 
        X,y = self.smote.fit_sample(X, y)
        return X, y.values.reshape(-1, 1)

    def transform(self, X):
        return X

class DF_OneHotEncoder(BaseEstimator, TransformerMixin):
    
    def __init__(self, filter_threshold = None):
        self.filter_threshold = filter_threshold

    def fit(self, X, y=None):
        self.onehot_cols = []
        for col in X.columns:
            if self.filter_threshold is not None:
                cat_inds = X[col].value_counts(normalize=True)>self.filter_threshold
                cat_cols = cat_inds.index[cat_inds].tolist()
                cat_cols = [f'{col}_{s}' for s in cat_cols]
                if len(cat_inds) == len(cat_cols): cat_cols = cat_cols[:-1]
                self.onehot_cols.extend(cat_cols)
        return self

    def transform(self, X):
        index = X.index
        for col in X.columns:
            one_hot = pd.get_dummies(X[col],prefix=col)
            X = X.drop(col,axis = 1)
            X = X.join(one_hot)
        X.reset_index(inplace = True,drop= True)
        if self.filter_threshold is not None:
            Dummy = pd.DataFrame(np.zeros((len(X),len(self.onehot_cols))), columns=self.onehot_cols)
            com_cols = Dummy.columns.intersection(X.columns)
            Dummy[com_cols] = X[com_cols]
            X = Dummy.set_index(index)
        return X

#class DF_LabelEncoder(BaseEstimator, TransformerMixin):
