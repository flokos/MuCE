import os
import sys
import json
import math

import numpy as np
import pandas as pd
#import seaborn as sns
import matplotlib.pylab as plt

from collections import deque
from bisect import bisect_left
from scipy.stats import rankdata
from scipy.stats import kendalltau
from scipy.optimize import linprog
from collections import OrderedDict
from scipy.interpolate import interp1d


    
def get_style():
            # Set the style globally
            # Alternatives include bmh, fivethirtyeight, ggplot,
            # dark_background, seaborn-deep, etc
            plt.style.use('seaborn-paper')
            plt.rcParams['figure.figsize'] = [6,4]
            plt.rcParams['figure.dpi'] = 300.0

            plt.rcParams['axes.labelsize'] = 18
            plt.rcParams['axes.titlesize'] = 16
            plt.rcParams['axes.edgecolor'] = 'black'
            plt.rcParams['axes.facecolor'] = 'white'
            plt.rcParams['axes.linewidth'] = 0.8 #set the value globally

            plt.rcParams['axes.xmargin'] = 0.0
            plt.rcParams['axes.ymargin'] = 0.0
            # set tick width
            plt.rcParams['xtick.major.size'] = 2

            plt.rcParams['xtick.minor.width'] = 0  
            plt.rcParams['xtick.direction'] = 'in'
            plt.rcParams['ytick.major.width'] = 1.1

            plt.rcParams['xtick.color'] = 'black'

            plt.rcParams['xtick.major.pad'] = 5.6
            plt.rcParams['ytick.major.pad'] = 5.6



            plt.rcParams['xtick.labelsize'] = 12
            plt.rcParams['ytick.labelsize'] = 12

            plt.rcParams['grid.linestyle'] ='--'
            plt.rcParams['grid.alpha'] = 0.2
            plt.rcParams['grid.color'] = 'black'
            plt.rcParams['grid.linewidth'] = 0.5        

def round_nearest(x, a):
    return round(round(x / a) * a, -int(math.floor(math.log10(a))))

import cProfile

class Survey:
    
    def __init__(self,filename,alpha):
        
        self.filename=filename
        self.alpha=alpha
        self.xlsdf1 = pd.read_excel(filename,0)
        self.xlsdf2 = pd.read_excel(filename,1)
        self.xlsdf3 = pd.read_excel(filename,2)
        
    
    def get_num_of_alternatives_survey(self):
        
        get_num_of_alternatives = len(self.xlsdf3)
        
        return get_num_of_alternatives
    
    def get_num_of_criteria_survey(self):    
    
        columnsnames = self.get_multicriteria_columns_survey()
        numofcrit = (len(columnsnames)/self.get_num_of_alternatives_survey())-1
        
        return numofcrit
    
    def get_multicriteria_columns_survey(self):
        
        columnsnames = self.get_semiM_survey().Code.values
        
        return columnsnames
    

    def get_semiM_survey(self):
        
        xlsx_df = self.xlsdf1
        
        semiM = xlsx_df[xlsx_df.Multicriteria == 1].reset_index(drop=True)
        
        return semiM
        
    def get_semiA_survey(self):
        
        columnsnames = self.get_multicriteria_columns_survey()
        
        semiA = self.xlsdf2.loc[:,columnsnames].abs()
        
        return semiA


    def create_criteria_data_survey(self):
        
        dict =  OrderedDict()
        cri = []
        mono = []
        typ = []
        worst = []
        best = []
        a = []
        
        numofcrit = self.get_num_of_criteria_survey()
        semiM = self.get_semiM_survey()
        indexlist = range(0,semiM.index[-1],numofcrit)
        
        for i in range(0,numofcrit):
            
            row = semiM.iloc[indexlist[i],:]
            
            cri.append("Criterion %d" %(i+1))
            
            mono.append(row.Type)
            
            if row.Alternatives==0:
                
                typ.append(0)
                
                if (row.Type==1) & (row.Min < 0) :
                    
                    worst.append(-1*row.Min)
                    
                    best.append(-1*row.Max)
                    
                    a.append(alpha)
                    
            else:
                
                typ.append(1)
                
                worst.append(row.Min)
                
                best.append(row.Max)
                
                a.append(row.Alternatives)
        
        dict['Cri_atributes']=cri
        dict['Monotonicity']=mono
        dict['Type']=typ
        dict['Worst']=worst
        dict['Best']=best
        dict['a']=a
        
        M = pd.DataFrame(dict)
        
        return M
        
    def get_alternatives_names(self):
        
        alternativesnames = self.xlsdf3[self.xlsdf3.columns[0]].tolist()
        
        return alternativesnames
    
    def get_criteria_names(self):
        
        if os.path.isfile(filename+'_metatable.csv'):
            metatable = pd.read_table(filename+'_metatable.csv', sep='\t')
        else:
            metatable = makeMetatable(filename, alpha)
            metatable.to_csv(filename+'_metatable.csv',sep='\t',index=False, encoding='utf-8')
        
        criterianames = metatable[metatable.columns[0]].tolist()
        
        return criterianames
        
    def create_users_data_survey(self):
        
        semiA = self.get_semiA_survey()
            
        alternativesnames = self.get_alternatives_names()
        
        numofalt = self.get_num_of_alternatives_survey()
        
        numofcrit = self.get_num_of_criteria_survey()
        
        criterianames = self.get_criteria_names()
        
        criterianames.append("Ranking")
        
        numofusers = len(semiA)
        
        namesofusers = []
        
        for user in range(1,len(semiA)+1):
            temp = "user%d" %user
            namesofusers.append(temp)
            
        iterables = [namesofusers, alternativesnames]
        
        index = pd.MultiIndex.from_product(iterables, names=['Users', 'Alt/cri'])
        
        peruserlist=[]
        
        for i in range(numofusers):
            
            position = 0
            positionlist=[]
            
            for c in range(numofalt):
                
                for j in range(numofcrit+1):
                    
                    if c==0:
                        
                        if j==0:
                            
                            positionlist.append(position)
                            
                        else:
                            
                            position = position+numofcrit
                            
                            positionlist.append(position)
                        
                
                peruserlist.append(semiA.iloc[i,positionlist].tolist())
                
                positionlist=[x+1 for x in positionlist]
            
        peruserdf = pd.DataFrame(peruserlist, index=index,columns=criterianames)
        
        return peruserdf


class IOtools:
    
    def __init__(self):
        
        self.filename = ""
        self.multidf = []
        self.onedf = []
        
    def write2excel_multiuser(self):
    
        writer = pd.ExcelWriter(self.filename+'_multiusers.xlsx', engine='xlsxwriter')
        
        self.multidf.to_excel(writer,'Data')
        
        writer.save()
        
        
    def write2csv_multiuser(self):
        
        self.multidf.to_csv(self.filename+'_multiusers.csv',sep='\t', encoding='utf-8')
        
        
    def load_from_csv_multiuser(self):
        
        csvmyltidf = pd.read_table(self.filename+'_multiusers.csv', sep='\t',index_col=[0,1] )
        
        return csvmyltidf
        
    def get_userdf(self,user):
       
        temp=self.multidf.index.get_level_values('Users').unique()
        
        multicriteria=self.multidf.loc[temp[user]]
        
        multicriteria1=multicriteria.reset_index(level=1)
        
        return multicriteria1
    
    def load_from_csv(self,filename1):
        
        with open(filename1) as f:
            
            table = pd.read_table(f, sep='\t')
        
        return table
        
    def write_to_csv(self,filename1):
        
        self.onedf.to_csv(filename1,sep='\t',index=False, encoding='utf-8')

class Utastar:
    
    def __init__(self,table,metatable,delta,epsilon,postopt=True):
        
        self.table=table
        self.metatable=metatable
        self.delta=delta
        self.epsilon=epsilon
        self.postopt=postopt
        
                
        names_of_alt=self.table[self.table.columns[0]]
        list_alt = names_of_alt.values
        self.alternatives = list_alt
        
        names_of_cri=self.table.columns.values
        list_cri=names_of_cri[1:len(names_of_cri)-1]
        self.criteria = list_cri
        
        self.intervals = OrderedDict()
        self.get_intervals()
        
        self.utilities_coeff_zeros = OrderedDict()
        self.utilities_coeff = OrderedDict()
        self.weights_coeff= OrderedDict()
        self.get_coeff()
        
        self.A = []
        self.Aeq = []
        self.bup = []
        self.beq = []
        self.f = []
        
        self.A_post = []
        self.bup_post = []
        self.f_post = OrderedDict()
        
        self.resopt = []
        self.opt_solution = []
        self.marginal_opt = [] 
        self.marginal_norm_opt = [] 
        self.model_weights_opt = []
        
        self.respost = OrderedDict()
        self.ftest = []
        self.post_solution_per_criterion = OrderedDict()
        self.post_solution = []
        self.marginal_post = [] 
        self.marginal_norm_post = [] 
        self.model_weights_post = []
        
        self.user_order=[]
        self.user_order = self.get_user_rank()
        
        self.global_utilities_opt = []
        self.global_utilities_post = []
        
        self.tau_opt = []
        self.tau_post = []
        
   
    def get_monotonicity(self,cri):
        
        cri_row = self.get_cri_row(cri)
        
        monotonicity = cri_row[cri_row.columns[1]].values
        
        return monotonicity[0]
    
    def get_type(self,cri):
        
        cri_row = self.get_cri_row(cri)
        
        type = cri_row[cri_row.columns[2]].values
        
        return type[0]
        
    
    def get_worst(self,cri):
        
        cri_row = self.get_cri_row(cri)
        
        worst = cri_row[cri_row.columns[3]].values
        
        return worst
    
    def get_best(self,cri):
        
        cri_row = self.get_cri_row(cri)
        
        best = cri_row[cri_row.columns[4]].values
        
        return best    
    
    def get_a(self,cri):
        
        cri_row = self.get_cri_row(cri)
        
        a = cri_row[cri_row.columns[5]].values
        
        return a[0]  
    
    def get_intervals(self):
        
        for cri in self.criteria:
            
            type = self.get_type(cri)
            worst = self.get_worst(cri)
            best = self.get_best(cri)
            a = self.get_a(cri)
            
            if type == 0:
                self.intervals[cri]=worst+(range(0,a)/(a-1.0))*(best-worst)
                self.intervals[cri]=self.intervals[cri].tolist()
            else:
                self.intervals[cri]=worst+(range(0,a)/(a-1.0))*(best-worst)
                self.intervals[cri]=self.intervals[cri].tolist()
                
    
    def get_intervals_index_zeros(self):
        l=[]
        for key,value in self.intervals.iteritems():
            for i in range(len(value)):
                l.append((key,value[i]))
        return l
        


    def get_intervals_index(self):
        l=[]
        for key,value in self.intervals.iteritems():
            for i in range(len(value)):
                if i!=0:
                    l.append((key,value[i]))                
        return l

    def get_cri_row(self,cri):

        return self.metatable.loc[self.metatable[self.metatable.columns[0]] == cri]


    def get_alt_row(self,alt):
        
        return self.table.loc[self.table[self.table.columns[0]] == alt]

    def get_value(self,cri,alt):
        
        alt_row = self.get_alt_row(alt)
        
        temp = alt_row[cri].values
        
        temp = 1.0*temp[0]
        
        return temp
        

    def get_position(self,cri,alt):
        
        monotonicity = self.get_monotonicity(cri)
        
        temp = self.get_value(cri,alt)
        
        testlenleft = len(self.intervals[cri])-1
        testposleft = np.digitize(temp,self.intervals[cri],right=False)
       
        testlenright = len(self.intervals[cri])-1
        testposright = np.digitize(temp,self.intervals[cri],right=True)
        
        if (monotonicity == 1) & (testlenleft == testposleft):
            return [testposleft-1, 1*testposleft]
            
        elif (monotonicity == 1) & (0 == testposleft):
            return [1*testposleft , testposleft+1]
        
        elif (monotonicity == 1) & (testlenleft != testposleft):
            return [testposleft-1, 1*testposleft]
            
        elif (monotonicity == 0) & (0 == testposright):
            return [1*testposright, testposright+1]
        
        elif (monotonicity == 0) & (testlenright == testposright):
            return [testposright-1, 1*testposright]
        
        elif (monotonicity == 0) & (testlenright != testposright):
            return [testposright-1, 1*testposright]
    
    
    def get_coeff(self):
        
        n=0
        for cri in self.criteria:
            
            for alt in self.alternatives:
               
                if n==0:
                    self.utilities_coeff_zeros[alt] = []
                    self.utilities_coeff[alt] = []
                    self.weights_coeff[alt] = []
                
                position = self.get_position(cri,alt)
                
                par_util_list=[0.0]*len(self.intervals[cri])
  
                x  = 1.0*self.get_value(cri,alt)
                x1 = 1.0*self.intervals[cri][position[1]]
                x0 = 1.0*self.intervals[cri][position[0]]
                
                slope = (x-x1)/(x0-x1)
                
                par_util_list[position[1]]=abs(1-slope)
                par_util_list[position[0]]=abs(slope)
                
                self.utilities_coeff_zeros[alt] =\
                self.utilities_coeff_zeros[alt]+par_util_list
                
                del par_util_list[0]
                
                self.utilities_coeff[alt] = \
                self.utilities_coeff[alt]+par_util_list
                
                weights=list(np.cumsum(par_util_list[::-1])[::-1])
                self.weights_coeff[alt] = self.weights_coeff[alt]+weights
                
            n=n+1

    def construct_lp(self):
        
        last_column=self.table.columns.values
        
        sorted_table =self.table.sort_values(last_column[-1])
        
        errors = [0.0]*((len(self.alternatives)-1)*2+2)
        errors[0]=-1.0
        errors[1]=1.0
        errors[2]=1.0
        errors[3]=-1.0
        derrors=deque(errors)
        
        names_list_list = sorted_table.iloc[:,[0]].values
        ranking_list_list = sorted_table.iloc[:,[-1]].values
        
        for i in range(1,len(ranking_list_list)):
            
            if (ranking_list_list[i-1][0] < ranking_list_list[i][0]):
                a = self.weights_coeff[names_list_list[i-1][0]]
                b = self.weights_coeff[names_list_list[i][0]]
                c = [a[i] - b[i] for i in range(len(a))]
                temp = c+list(derrors)
                temp1 = [ -x for x in temp]
                self.A.append(temp1)
                derrors.rotate(2)
                self.bup.append(-1.0*self.delta)
            elif (ranking_list_list[i-1][0] == ranking_list_list[i][0]):
                a = self.weights_coeff[names_list_list[i-1][0]]
                b = self.weights_coeff[names_list_list[i][0]]
                c = [a[i] - b[i] for i in range(len(a))]
                temp = c+list(derrors)
                temp1 = [ -x for x in temp]
                self.Aeq.append(temp1)
                derrors.rotate(2)
                self.beq.append(0.0)
        
        sumw=[1.0]*len(a) 
        temp2=[0.0]*((len(self.alternatives)-1)*2+2)
        sumw= sumw+temp2
        temp3=[0.0]*len(a) 
        temp4=[1.0]*((len(self.alternatives)-1)*2+2)
        
        self.Aeq.append(sumw)
        self.beq.append(1)
        self.f=temp3+temp4
        
    
    def construct_post_lps(self,sumoferrorsopt):
        
        self.A_post =[] + self.A
        self.A_post.append(self.f)
        self.bup_post = [] + self.bup
        self.bup_post.append(sumoferrorsopt+self.epsilon)
        
        criteria_fpost_zeros={}
        criteria_fpost_ones={}
        
        for cri in self.criteria:
            
            a = self.get_a(cri)
            
            temp1=[0.0]*(a-1)
            temp2=[1.0]*(a-1)
            temp3=[ -x for x in temp2]
            criteria_fpost_zeros[cri]=temp1
            criteria_fpost_ones[cri]=temp3
            
        for cri in self.criteria:
            
            crisecond=cri
            
            fpost=[]
            
            for crifirst in self.criteria:
                
                if crifirst==crisecond:
                    fpost=fpost+criteria_fpost_ones[crifirst]
                else:
                    fpost=fpost+criteria_fpost_zeros[crifirst]
            self.ftest=fpost
            fpost=fpost+[1.0]*((len(self.alternatives)-1)*2+2)
            self.f_post[cri]=fpost
    
    def get_solution(self,object):
        
        x = list(object.x[0:len(self.weights_coeff[self.alternatives[0]])])
            
        return x
    
    def construct_u_g(self,xfinal):
        
        marginal=OrderedDict()
        marginal_norm=OrderedDict()
        model_weights=OrderedDict()
        n = 0
        for cri in self.criteria:
            
            temp_list_util=[]
            temp_list_util.append(0)
            for i in range(n,n+len(self.intervals[cri])-1):
                
                temp_list_util.append(xfinal[i])
                n=n+1
            acum=np.cumsum(temp_list_util)
            temp=max(acum)
            if temp>0:
                acumnorm = [ x/temp for x in acum]
            else:
                acumnorm = acum
            
            marginal[cri]=list(acum)
            marginal_norm[cri]=list(acumnorm)
            model_weights[cri]=temp
            
        return marginal ,marginal_norm, model_weights
    
    
    def get_user_rank(self):
        
        UserRank = []
        names_of_cri=self.table.columns.values
        list_rank = names_of_cri[len(names_of_cri)-1]
        for alt in self.alternatives:
            alt_row = self.get_alt_row(alt)
            # Upost[alt]=np.dot(par_weights_matrix[alt], final_sol_post)
            # Urank.append(U[alt])
            #print alt
            temp = list(alt_row[list_rank].values)
            UserRank.append(temp[0])
        ru = rankdata(UserRank, method='dense')
        return ru
        
    def get_global_utilities(self,object):
        
        U = OrderedDict()
        
        for alt in self.alternatives:
            
            U[alt]=np.dot(self.weights_coeff[alt], object)
        
        return U
    
    
    def get_tau_kendall(self,obj1,obj2):
        
        tau, p = kendalltau(obj1, obj2)
        
        return tau
        
        
    def get_global_utilities_list(self,object):
        
        U = []
        
        for alt in self.alternatives:
            
            U.append(np.dot(self.weights_coeff[alt], object))
        
        return U

    def solve(self):
        
        self.construct_lp()
        
        self.resopt = linprog(self.f, self.A, self.bup, self.Aeq, self.beq,\
                      method='simplex')
        
        sumoferrorsopt=self.resopt.fun
        self.opt_solution = self.get_solution(self.resopt)
        self.marginal_opt,self.marginal_norm_opt,self.model_weights_opt = \
        self.construct_u_g(self.opt_solution)
        
        self.global_utilities_opt = self.get_global_utilities(self.opt_solution)
        tempU = self.get_global_utilities_list(self.opt_solution)
        tempU1 = [ round_nearest(-x,self.epsilon) for x in tempU]
        
        temptauopt = self.get_tau_kendall(rankdata(tempU1, method='dense'),self.user_order)
        if temptauopt == 1.0:
            self.tau_opt = temptauopt
        elif temptauopt > 1.0:
            self.tau_opt = temptauopt-sys.float_info.epsilon
        elif temptauopt < 1.0:
            self.tau_opt = temptauopt+sys.float_info.epsilon
        
        #print self.tau_opt
        
        temp_array =[]
        if self.postopt:
            xfinals={}
            list_xfinals=[]
            self.construct_post_lps(sumoferrorsopt)
            for cri in self.criteria:
                
                self.respost[cri] = linprog(self.f_post[cri], self.A_post,\
                        self.bup_post,self.Aeq, self.beq, method='simplex')
                
                self.post_solution_per_criterion[cri] = \
                self.get_solution(self.respost[cri])
                temp_array.append(self.post_solution_per_criterion[cri])
            
            self.post_solution=[sum(i)/len(self.criteria) for i in zip(*temp_array)]
            
            self.marginal_post, self.marginal_norm_post, self.model_weights_post =\
            self.construct_u_g(self.post_solution)

            self.global_utilities_post = self.get_global_utilities(self.post_solution)
            tempU = self.get_global_utilities_list(self.post_solution)
            tempU1 = [ round_nearest(-x,self.epsilon) for x in tempU]

            temptauopt = self.get_tau_kendall(rankdata(tempU1, method='dense'),self.user_order)
            if temptauopt == 1.0:
                self.tau_post = temptauopt
            elif temptauopt > 1.0:
                self.tau_post = temptauopt-sys.float_info.epsilon
            elif temptauopt < 1.0:
                self.tau_post = temptauopt+sys.float_info.epsilon

   
    def plot_model_weights(self):
        
        def get_style():
                    # Set the style globally
                    # Alternatives include bmh, fivethirtyeight, ggplot,
                    # dark_background, seaborn-deep, etc
                    plt.style.use('seaborn-paper')
                    plt.rcParams['figure.figsize'] = [6,4]
                    plt.rcParams['figure.dpi'] = 300.0

                    plt.rcParams['axes.labelsize'] = 18
                    plt.rcParams['axes.titlesize'] = 16
                    plt.rcParams['axes.edgecolor'] = 'black'
                    plt.rcParams['axes.facecolor'] = 'white'
                    plt.rcParams['axes.linewidth'] = 0.8 #set the value globally

                    plt.rcParams['axes.xmargin'] = 0.0
                    plt.rcParams['axes.ymargin'] = 0.0
                    # set tick width
                    plt.rcParams['xtick.major.size'] = 2

                    plt.rcParams['xtick.minor.width'] = 0  
                    plt.rcParams['xtick.direction'] = 'in'
                    plt.rcParams['ytick.major.width'] = 1.1

                    plt.rcParams['xtick.color'] = 'black'

                    plt.rcParams['xtick.major.pad'] = 5.6
                    plt.rcParams['ytick.major.pad'] = 5.6



                    plt.rcParams['xtick.labelsize'] = 12
                    plt.rcParams['ytick.labelsize'] = 12

                    plt.rcParams['grid.linestyle'] ='--'
                    plt.rcParams['grid.alpha'] = 0.2
                    plt.rcParams['grid.color'] = 'black'
                    plt.rcParams['grid.linewidth'] = 0.5


        get_style()

        fig = plt.figure()
        ax = fig.gca()


        ax.bar(range(len(self.model_weights_post))[::-1], self.model_weights_post.values(), align='center',color='grey',alpha=0.8)
        plt.xticks(range(len(self.model_weights_post))[::-1], self.model_weights_post.keys())
        plt.title('Model weights')


        plt.tight_layout()

def realign_polar_xticks(ax):
    for theta, label in zip(ax.get_xticks(), ax.get_xticklabels()):
        theta = theta * ax.get_theta_direction() + ax.get_theta_offset()
        theta = np.pi/2 - theta
        y, x = np.cos(theta), np.sin(theta)
        if x >= 0.1:
            label.set_horizontalalignment('left')
        if x <= -0.1:
            label.set_horizontalalignment('right')
        if y >= 0.5:
            label.set_verticalalignment('bottom')
        if y <= -0.5:
            label.set_verticalalignment('top')

def _invert(x, limits):
    """inverts a value x on a scale from
    limits[0] to limits[1]"""
    return limits[1] - (x - limits[0])

def _scale_data(data, ranges):
    """scales data[1:] to ranges[0],
    inverts if the scale is reversed"""
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        try:
            assert (y1 <= d <= y2) or (y2 <= d <= y1)
        except AssertionError:
            print(d,y1,y2)
    x1, x2 = ranges[0]
    d = data[0]
    if x1 > x2:
        d = _invert(d, (x1, x2))
        x1, x2 = x2, x1
    sdata = [d]
    for d, (y1, y2) in zip(data[1:], ranges[1:]):
        if y1 > y2:
            d = _invert(d, (y1, y2))
            y1, y2 = y2, y1
        sdata.append((d-y1) / (y2-y1) 
                     * (x2 - x1) + x1)
    return sdata

class ComplexRadar():
    def __init__(self, fig, variables, ranges,
                 n_ordinate_levels=6):
        angles = np.arange(0, 360, 360./len(variables))

        axes = [fig.add_axes([0.1,0.1,0.8,0.8],polar=True,
                label = "axes{}".format(i)) 
                for i in range(len(variables))]
        l, text = axes[0].set_thetagrids(angles, 
                                         labels=variables)
        [txt.set_rotation(angle-90) for txt, angle 
             in zip(text, angles)]
        
        for ax in axes[1:]:
            ax.patch.set_visible(False)
            ax.grid("off")
            ax.xaxis.set_visible(False)
            
        for i, ax in enumerate(axes):
            grid = np.linspace(*ranges[i], 
                               num=n_ordinate_levels)
            gridlabel = ["{}".format(round(x,2)) 
                         for x in grid]
            if ranges[i][0] > ranges[i][1]:
                grid = grid[::-1] # hack to invert grid
                          # gridlabels aren't reversed
            gridlabel[0] = "" # clean up origin
            ax.set_rgrids(grid, labels=gridlabel,
                         angle=angles[i])
            ax.spines["polar"].set_visible(False)
            ax.set_ylim(*ranges[i])
        # variables for plotting
        self.angle = np.deg2rad(np.r_[angles, angles[0]])
        self.ranges = ranges
        self.ax = axes[0]
        realign_polar_xticks(ax)
    def plot(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.plot(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
    def fill(self, data, *args, **kw):
        sdata = _scale_data(data, self.ranges)
        self.ax.fill(self.angle, np.r_[sdata, sdata[0]], *args, **kw)
