import os
import pandas as pd
from sklearn import model_selection

'''
This code is refactored form 
https://github.com/abhi1thakur/mlframework/blob/master/src/cross_validation.py
'''

class CrossValidation:

    def __init__(self):

        self.df = pd.read_csv(os.environ.get('DATA_PATH'))
        self.problem_type = os.environ.get('PROBLEM_TYPE')
        self.shuffle = os.environ.get('SHUFFLE')
        self.kfolds = int(os.environ.get('KFOLDS'))
        self.random_state = 42
        self.target_cols = os.environ.get('TARGET_COLS').split()
        self.multilabel_delimiter = os.environ.get('MULTI_DELIMITER')
        self.num_targets = len(self.target_cols)
        self.save_path = os.environ.get('SAVE_PATH')

        if(self.shuffle):
            self.df = self.df.sample(frac=1).reset_index(drop=True)

        self.df['kfolds'] = -1

    

    def  _stratifiedKfold_regression(self):

        ''' 
        sort the values for target value and then get each chunk of size k
        and assign each row with k-fold value indexed 0

        for samples // k_folds, complete k-fold sequence assignment takes place
        samples % kfolds samples takes value of k upto the remainder length 
        '''

        num_samples = len(self.df)
        self.df = self.df.sort_values(self.target_cols,ascending=True).reset_index(drop=True)
        for i in range(0,num_samples,self.kfolds):
            k_counter = 0
            for j in range(i,min(i+self.kfolds,num_samples)):
                self.df.loc[j,'kfolds'] = k_counter
                k_counter +=1

    def _binary_multiclass(self):

        if(self.num_targets != 1):
            raise Exception('Invalid number of targets')

        target = self.target_cols[0]
        unique_val = self.df[target].nunique()

        if(unique_val == 1):
            raise Exception('Only one unique value found')

        elif(unique_val > 1):
            kf = model_selection.StratifiedKFold(n_splits=self.kfolds,random_state=self.random_state,shuffle=False)

            for fold,(_,val_idx) in enumerate(kf.split(X=self.df,y=self.df[target].values)):
                self.df.loc[val_idx,'kfolds'] = fold
        else:
            raise Exception('Problem Not Understood')

    
    def _kfold_single_multicols_regression(self):

        if(self.num_targets != 1 and self.problem_type == 'single_col_regression'):
            raise Exception('Invalid number of targets for this problem')

        if(self.num_targets < 2 and self.problem_type('multi_col_regression')):
            raise Exception('Targets less than required for multi regression problem ')

        kf = model_selection.KFold(n_splits=self.kfolds)
        for fold, (_,val_idx) in enumerate(kf.split(X=self.df)):
            self.df.loc[val_idx,'kfolds'] = fold

        
    def _holdout(self,holdout_pct):

        if(self.num_targets != 1):
            raise Exception('Invalid targets found for this problem type')

        num_holdout_samples = int(len(self.df) * holdout_pct / 100)
        self.df.loc[:len(self.df)-num_holdout_samples,'kfolds'] = 0
        self.df.loc[len(self.df)-num_holdout_samples:,'kfolds'] = 1


    def _multilabel_classification(self):

        if self.num_targets != 1:
                raise Exception("Invalid number of targets for this problem type")

        targets = self.df[self.target_cols[0]].apply(lambda x: len(str(x).split(self.multilabel_delimiter)))
        kf = model_selection.StratifiedKFold(n_splits=self.kfolds)
        for fold, (_, val_idx) in enumerate(kf.split(X=self.df, y=targets)):
            self.df.loc[val_idx, 'kfolds'] = fold
        

    def _printing(self):

        print(self.df.kfolds.value_counts())

        #used only for classification
        '''
        for fold in range(self.kfolds):
            print(f'Fold {fold}')
            print(self.df[self.df['kfolds'] == fold][self.target_cols[0]].value_counts())
        '''

    def split(self):

        
        if(self.problem_type in ('binary_classification','multiclass_classification')):
            self._binary_multiclass()
        
        elif(self.problem_type in ('single_col_regression','mulit_col_regression')):
            self._stratifiedKfold_regression()
    
        elif(self.problem_type.startswith('holdout_')):
            holdout_pct = int(self.problem_type.split("_")[1])
            self._holdout(holdout_pct)

        elif(self.problem_type == 'multilabel_classification'):
            self._multilabel_classification()

        else:
            raise Exception('Problem Not Understood')

        self._printing()
        self.df.to_csv(self.save_path)

        return self.df

        
if __name__ == '__main__':

    cv = CrossValidation()
    df = cv.split()









