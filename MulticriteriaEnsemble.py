from sklearn.model_selection import train_test_split
from mlxtend.classifier import EnsembleVoteClassifier
import numpy as np
import pandas as pd
from utalib import IOtools, Utastar
from utalib import *
import seaborn as sns
import matplotlib.pylab as plt
import matplotlib
from collections import OrderedDict
import scipy
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.externals import joblib
from pso import Particle
from pso import PSO as psopt
import shutil
import os
import datetime
import re
import jenkspy

class MulticriteriaEnsemble(object):
    def __init__(self,models=OrderedDict({}),dataset=None,pickle_path=None,crit_metrics=None,global_metric=None,delta=None,epsilon=None,a=None,bootstrap_models=OrderedDict({}),n_splits=5,voting='soft',jenks=True,jenks_limit=2,refit=False):
        self.models=models
        self.bootstrap_models = bootstrap_models
        self.dataset = dataset
        self.crit_metrics=crit_metrics
        self.global_metric=global_metric
        self.delta = delta
        self.best_delta = None
        self.epsilon = epsilon
        self.a = a
        self.voting = voting
        self.n_splits = n_splits
        self.refit = refit
        self.pickle_path = self.dataset.path+'base_learners/'
        self.multicriteria_table = None
        self.meta_table = None
        self.utastar_model = None
        self.wmv_model = None
        self.natural_breaks = None
        self.weights = []
        self.global_utilities = []
        self.kfold_indices = []
        self.test_kfold_indices = []
        self.global_metrics = []
        self.is_fit = {
            'wmv':False,
            'clfs':not self.refit,
            'utastar':False,
        }
        self.jenks = jenks
        self.jenks_limit = jenks_limit
        if not self.models and refit==True:
            raise Exception('Base learners are not provided.')
        elif self.models and refit==False:
            raise Exception('Models parameter should not be set to anything while refit=False')
        if self.dataset == None:
            raise Exception('Dataset is not provided.')
        if self.crit_metrics == None:
            raise Exception('Performance estimators are not provided.')
        if self.global_metric == None:
            raise Exception('Global Performance estimator is not provided.')
        if self.delta == None or self.a == None or self.epsilon == None:
            raise Exception('One or more utastar model parameters is/are not provided.')
    
    def _pso_cost(self,x):
        self.delta = x[0]
        self.epsilon = x[1]
        if self.is_fit['wmv']:
            self.fit(mtable=False)
        else:
            self.fit()
        return 1-self.score()
    
    def pso(self,bounds,num_particles,w,c1,c2,maxiter,threshold):
        psopt(self._pso_cost,bounds,num_particles,w,c1,c2,maxiter,threshold)
        
        
        
    def _save_model(self,model,file_name):
        print "Saving Model!"
        if os.path.isfile(self.pickle_path+file_name):
            if not os.path.exists(self.pickle_path+'Archive/'):
                os.makedirs(self.pickle_path+'Archive/') 
            archived_file_name = self.pickle_path+'Archive/'+file_name.replace('.pkl','_')+datetime.datetime.today().strftime("%m-%d-%Y-%H%M%S")+'.pkl'
            shutil.move(self.pickle_path+file_name,archived_file_name)
            joblib.dump(model,self.pickle_path+file_name)
            print "Model Saved!!!"
        else:
            print "Model Saved!!!"
            joblib.dump(model,self.pickle_path+file_name)
                    
                    
                    
        
    #Reinitialize crucial variables
    def _reset(self):
        self.global_utilities = []
        self.weights = []
        self.kfold_indices = []
        if self.refit == True:
            self.bootstrap_models = OrderedDict({})
            print 'Multicriteria Table Deleted!!!'
            self.multicriteria_table = None
            self.meta_table = None
            
    #Split dataset to k stratified folds and save the indices
    def _skfold(self,n_splits):
        skf = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=12345)
        for train_index,test_index in skf.split(self.dataset.X_train,self.dataset.y_train):
            self.kfold_indices.append(train_index.tolist())
            self.test_kfold_indices.append(test_index.tolist())
    #Fit the base learners
    def _fit_clfs(self):
        #If the path that the models will be saved does not exist create it
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        #For every fold 
        for k_idx,k in enumerate(self.kfold_indices):
            #Make a copy of the base learners
            temp_models = OrderedDict(zip(self.models.keys(),clone(self.models.values())))
            #For every model in the base learners create a separate model , train it on the current fold and save it 
            for model in temp_models.keys():
                model_name = '%s_%s_FOLD%i'%(model.replace('_'+self.dataset.name,''),self.dataset.name,k_idx)
                temp_models[model].fit(self.dataset.X_train.iloc[k],self.dataset.y_train.iloc[k])
                file_name = model_name+'.pkl'
                self._save_model(temp_models[model],file_name)
                self.bootstrap_models[model_name] = temp_models[model]
                
        #Rename the base learners to include the dataset name,fit the models and save them
        if not self.dataset.name in model:
            self.models = self._rename_models(self.models)
        for model in self.models.keys():
            self.models[model].fit(self.dataset.X_train,self.dataset.y_train)
            self._save_model(self.models[model],model+'.pkl')
    #Fit the utastar model
    def _fit_utastar(self):
        #Define the Utastar model
        self.utastar_model = Utastar(self.multicriteria_table,self.meta_table,self.delta,self.epsilon)
        #Fit the Utastar model
        self.utastar_model.solve()    
    
    def _get_global_utilities(self):
        metrics = self._get_metrics(self.bootstrap_models,on='test')
        self._utastar_predict(metrics)
        
    #Fit the Weighted Majority Voting model
    def _fit_wmv(self):
        #Merge the base learners and the produced models(extra)
        models = self.bootstrap_models.values()
        #Define the Weighted Majority Voting model
        self.wmv_model = EnsembleVoteClassifier(clfs=models, weights=self.weights, voting=self.voting, refit=False)
        #Fit the WMV model
        self.wmv_model.fit(self.dataset.X_train,self.dataset.y_train)
    #Fit the Multicriteria Ensemble Model
    def fit(self,mtable = True):
        #Reinitilize crucial variables
        self._reset()
        #Get Stratified K-Fold indices
        self._skfold(self.n_splits)
        #if refit is needed,fit the models
        if self.refit:
            self._fit_clfs()
            self.is_fit['clfs'] = True
        else:
            #Check if
            try:
                for base_learner in os.walk(self.pickle_path).next()[2]:
                    if 'FOLD' in base_learner:
                        self.bootstrap_models[base_learner.replace('.pkl','')] = joblib.load(self.pickle_path+'%s'%base_learner) 
                    else:
                        self.models[base_learner.replace('.pkl','')] = joblib.load(self.pickle_path+'%s'%base_learner) 
                dummy_var = self.bootstrap_models.keys()[1]
            except:
                raise AttributeError('Refit is set to False but no models are given.')
        if mtable == False and self.multicriteria_table is None:
            raise Exception('Multicriteria table not found.Please run fit(mtable=True) at least once.')
        elif mtable == True:
            print 'Multicriteria table formed!!!'
            self._get_meta_table()
            self._get_multicriteria_table()
        self._fit_utastar()
        self._get_global_utilities()
        self._get_clfs_weights()
        self._fit_wmv()
        self.is_fit['wmv'] = True
    def predict(self,X):
        return self.wmv_model.predict(X)
    
    def predict_proba(self,X):
        return self.wmv_model.predict_proba(X)

    def _get_clfs_weights(self):
        gu = self.global_utilities
        if self.jenks == True:
            self.natural_breaks = jenkspy.jenks_breaks(gu, nb_class=5)
            gu = [i if i >= self.natural_breaks[-self.jenks_limit] else 0 for i in gu]
        gu_sum = sum(gu)
        for value in gu:
            self.weights.append(value/gu_sum)
            

    def add_clfs(self,clfs,refit=False):
        clfs = self._rename_models(clfs)
        if set(self.models.keys()).isdisjoint(clfs.keys()):
            if not refit:
                metrics = self._get_metrics(clfs)
                self.models.update(clfs)
            else:
                temp_models = {}
                for clf in clfs.keys():
                    temp_models[clf] = clone(clfs[clf])
                    temp_models[clf].fit(self.dataset.X_train,self.dataset.y_train)
                metrics = self._get_metrics(temp_models)
                self.models.update(temp_models)
            self._utastar_predict(metrics)
            self.weights = []
            self._get_clfs_weights()
            self._fit_wmv()
        else:
            raise Exception('One or more models are already in the ensemble.')

    def score(self):
        return self._get_global_metrics({'wmv':self.wmv_model},on='test')[0]

    def _utastar_predict(self,metrics):
        for clf_metrics in metrics:
            pred_partial_util = []
            for crit in self.utastar_model.criteria:
                X = self.utastar_model.intervals[crit]
                y = self.utastar_model.marginal_post[crit]
                pred_partial_util.append(np.interp(clf_metrics[self.utastar_model.criteria.tolist().index(crit)+1],X,y))
            pred_global_util = np.array(pred_partial_util).dot(np.array(clf_metrics[1:]))
            self.global_utilities.append(pred_global_util)

    def _rename_models(self,models):
        for model in models.keys():
                model_name = '%s_%s'%(model,self.dataset.name)
                models[model_name] = models.pop(model)
        return models

    def plot_partial_utilities(self):
        numofcriteria =len(self.utastar_model.criteria)
        n = numofcriteria

        if n % 2 == 0:
            fig1, axs = plt.subplots(n/2, 2,figsize=(18, 18))
        else:
            fig1, axs = plt.subplots(n/2+1, 2,figsize=(18, 18))
        for i in range(n):
            y = self.utastar_model.marginal_post[self.utastar_model.criteria[i]]
            x = self.utastar_model.intervals[self.utastar_model.criteria[i]]
            if i % 2 == 0:
                if self.utastar_model.get_type(self.utastar_model.criteria[i])==1:
                    axs[i/2, 0].plot(x, y, '--ok')
                    axs[i/2, 0].set_title(self.utastar_model.criteria[i])
                    axs[i/2, 0].set_xticks(x)
                    axs[i/2, 0].set_xlim(x[0],x[-1])
                    axs[i/2, 0].set_ylabel(r'$u_{%d}(g_{%d})$'%((i+1),(i+1)))
                    axs[i/2, 0].yaxis.grid(False)
                    if self.utastar_model.get_monotonicity(self.utastar_model.criteria[i])==1:
                        axs[i/2, 0].set_xlim(x[-1],x[0])
                else:
                    axs[i/2, 0].plot(x, y, '-ok')
                    axs[i/2, 0].set_title(self.utastar_model.criteria[i])
                    axs[i/2, 0].set_xticks(x)
                    axs[i/2, 0].set_xlim(x[0],x[-1])
                    axs[i/2, 0].set_ylabel(r'$u_{%d}(g_{%d})$'%((i+1),(i+1)))
                    axs[i/2, 0].yaxis.grid(False)
                    if self.utastar_model.get_monotonicity(self.utastar_model.criteria[i])==1:
                        axs[i/2, 0].set_xlim(x[-1],x[0])
            else:
                if self.utastar_model.get_type(self.utastar_model.criteria[i])==1:
                    axs[i/2, 1].plot(x, y, '--ok')
                    axs[i/2, 1].set_title(self.utastar_model.criteria[i])
                    axs[i/2, 1].set_xticks(x)
                    axs[i/2, 1].set_xlim(x[0],x[-1])
                    axs[i/2, 1].set_ylabel(r'$u_{%d}(g_{%d})$'%((i+1),(i+1)))
                    axs[i/2, 1].yaxis.grid(False)
                    if self.utastar_model.get_monotonicity(self.utastar_model.criteria[i])==1:
                        axs[i/2, 1].set_xlim(x[-1],x[0])

                else:
                    axs[i/2, 1].plot(x, y, '-ok')
                    axs[i/2, 1].set_title(self.utastar_model.criteria[i])
                    axs[i/2, 1].set_xticks(x)
                    axs[i/2, 1].set_xlim(x[0],x[-1])
                    axs[i/2, 1].set_ylabel(r'$u_{%d}(g_{%d})$'%((i+1),(i+1)))
                    axs[i/2, 1].yaxis.grid(False)
                    if self.utastar_model.get_monotonicity(self.utastar_model.criteria[i])==1:
                        axs[i/2, 1].set_xlim(x[-1],x[0])
        if n % 2 != 0:
            for l in axs[i/2-1,1].get_xaxis().get_majorticklabels():
                l.set_visible(True)
            fig1.delaxes(axs[i/2, 1])
        #plt.subplots_adjust(wspace = 0.3,hspace = 0.3)
        plt.tight_layout()
        plt.show()

    def plot_global_utilities(self):
        fig4 = plt.figure(4)
        ax = fig4.gca()
        ax.barh(range(len(self.utastar_model.global_utilities_post))[::-1], self.utastar_model.global_utilities_post.values(), align='center',color='grey',alpha=0.8)
        plt.yticks(range(len(self.utastar_model.global_utilities_post))[::-1], self.utastar_model.global_utilities_post.keys())
        ax.plot(self.utastar_model.global_utilities_post.values(),range(len(self.utastar_model.global_utilities_post))[::-1],linestyle='--',color='black',alpha=0.8)
        plt.xlim(0,1)
        plt.title('Ranking')
        plt.tight_layout()
        plt.show()
    
    def plot_global_utilities_pred(self):
        fig4 = plt.figure(4)
        ax = fig4.gca()
        ax.barh(range(len(self.global_utilities))[::-1], self.global_utilities, align='center',color='grey',alpha=0.8)
        plt.yticks(range(len(self.global_utilities))[::-1], self.bootstrap_models.keys())
        ax.plot(self.global_utilities,range(len(self.global_utilities))[::-1],linestyle='--',color='black',alpha=0.8)
        plt.xlim(0,1)
        plt.title('Ranking')
        plt.tight_layout()
        plt.show()
    
    def plot_criteria_weights(self):
        variables = self.utastar_model.model_weights_post.keys()
        data = self.utastar_model.model_weights_post.values()
        ranges = [(0.00001, 0.00001+max(self.utastar_model.model_weights_post.values()))]*len(self.utastar_model.criteria)
        fig1 = plt.figure(figsize=(10, 10))
        radar = ComplexRadar(fig1, variables, ranges,7)
        radar.plot(data)
        #dradar.fill(data, alpha=0.2, color='grey')
        plt.show()
    
    def plot_model_weights(self,title):
        sns.set(style="whitegrid")
        f, ax = plt.subplots(figsize=(10, 4))
        variables = dict(sorted(zip(self.bootstrap_models.keys(),self.weights)))
        sns.set_color_codes("pastel")
        f = sns.barplot(x=variables.keys(), y=variables.values(), color="b").set_title(title)
        ax.set_xticklabels(ax.get_xticklabels(),rotation=45,fontdict={
                'verticalalignment': 'baseline',
                'horizontalalignment': 'right'})
        ax.set(xlim=(-1, 30), ylabel="Weight",xlabel="Models")
        sns.despine(left=True, bottom=True)
        
    def _get_meta_table(self):
        columns = ['Cri/atributes','Monotonicity','Type','Worst','Best','a']
        meta_table = []
        for metric in self.crit_metrics.keys():
            monotonicity = 1
            if self.crit_metrics[metric][0]._sign == -1:
                monotonicity = 0
                self.crit_metrics[metric][0]._sign = 1
            mt_metric = [metric,monotonicity,0,self.crit_metrics[metric][1],self.crit_metrics[metric][2],self.a]
            meta_table.append(mt_metric)
        self.meta_table = pd.DataFrame(meta_table,columns=columns)

    def _get_multicriteria_table(self):
        criteria = self.crit_metrics.keys()
        columns = ['Alt/Cri ']
        columns.extend(criteria)
        #metrics_orig = self._get_metrics(self.models)
        metrics_bootstrap = self._get_metrics(self.bootstrap_models,on='validation')
        metrics = metrics_bootstrap
        multicriteria_table = pd.DataFrame(metrics,columns=columns)
        ranking = self._get_init_ranking()
        ranking = pd.DataFrame(ranking,columns=['Ranking'])
        self.multicriteria_table = multicriteria_table.join(ranking).copy(deep=True) 
     
                                       
    def _get_dataset(self,model,on='test'):
        if on == 'test':
            X,y = self.dataset.X_test.copy(),self.dataset.y_test.copy()
        elif on == 'validation':
            X,y = self.dataset.X_train.copy(),self.dataset.y_train.copy()
            if 'FOLD' in model:
                fold_idx = int(re.search(r'(?<=FOLD)[0-9]',model).group(0))
                indices = self.test_kfold_indices[fold_idx]
                X,y = X.iloc[indices],y.iloc[indices]
        elif on == 'train':
            X,y = self.dataset.X_train,self.dataset.y_train
            if 'FOLD' in model:
                fold_idx = int(re.search(r'(?<=FOLD_)[0-9]',model).group(0))
                indices = self.kfold_indices[fold_idx]
                X,y = X.iloc[indices],y.iloc[indices]
        else:
            raise Exception('Unexpected input for argument on.')
        return X,y
    
                                       
    def _get_global_metrics(self,models,on='test'):
        global_metrics = []
        for model in models.keys():
            X,y = self._get_dataset(model,on=on)
            global_metrics.append(self.global_metric(models[model],X,y))
        return global_metrics
    
    def _get_init_ranking(self):
        #gm_orig = self._get_global_metrics(self.models,on='validation')
        gm_bootstrap = self._get_global_metrics(self.bootstrap_models,on='validation')
        #gm = gm_orig + gm_bootstrap
        self.global_metrics = gm_bootstrap
        if self.global_metric._sign == 1:
            ranking = len(self.global_metrics) - scipy.stats.rankdata(self.global_metrics,method='max')
        else:
            ranking = scipy.stats.rankdata(gm,method='max')
        return ranking

    def _get_metrics(self,models,on='test'):
        metrics = []
        for model in models.keys():
            model_metrics = [model]
            X,y = self._get_dataset(model,on=on)
            for metric in self.crit_metrics.keys():
                mes = self.crit_metrics[metric][0](models[model],X,y)
                #Takes care of the negativde values on the multicriteria table and replaces them with 0
                if mes > 0:
                    model_metrics.append(mes)
                else:
                    model_metrics.append(0)
            metrics.append(model_metrics)
        return metrics