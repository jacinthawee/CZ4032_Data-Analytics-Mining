from read import read
from pre_processing import pre_process
from cba_rg import rule_generator
from cba_cb_m1 import M1
from cba_cb_m1 import cover
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.base import BaseEstimator

class M1WithoutPrune(BaseEstimator):

    def __init__(self):
        self.minsup = 0.01
        self.minconf = 0.5
        self.classifier = None
        self.cba_rg_time=0
        self.m1_time=0

    def fit(self, X_train, y_train, sample_weight=None):
        dataset = X_train
        dataset = dataset.tolist()
        for i in range(len(y_train)):
            dataset[i].append(y_train[i])
        start_time = time.time()
        cars = rule_generator(dataset, self.minsup, self.minconf,False)
        end_time = time.time()
        cba_rg_runtime = end_time - start_time

        start_time = time.time()
        classifier_m1 = M1(cars, dataset,False)
        end_time = time.time()
        cba_cb_runtime = end_time - start_time
        
        self.cba_rg_time=cba_rg_runtime+self.cba_rg_time
        self.m1_time=cba_cb_runtime+self.m1_time

        self.classifier = classifier_m1
    
    def predict(self, X_test):
        dataset = X_test
        preds = []
        for data in dataset:
            test_cond_set = []
            for column in range(0, len(data)-1):
                cond_set = [column, data[column]]
                test_cond_set.append(cond_set)
            potential = []
            for rule in self.classifier.rule_list:
                condset_list, classlabel = rule.get_condset()
                for condset in condset_list:
                    to_add = [condset_list, classlabel]
                    for i in test_cond_set:
                        if condset == i:
                            potential.append(to_add)
            idx_to_remove = []
            idx = 0
            for p in potential:
                for element in p[0]:
                    for i in test_cond_set:
                        if i[0] == element[0] and i[1] != element[1]:
                            idx_to_remove.append(idx)
                idx += 1
            all_idx = list(range(len(potential)))
            idx_to_keep = list(set(all_idx) - set(idx_to_remove))
            new_potential = []
            for i in idx_to_keep:
                new_potential.append(potential[i])
            if len(new_potential) > 1: # in case more than one potential matches, take majority
                multiple_preds = {}
                for p in new_potential:
                    pred = p[1]
                    if pred in multiple_preds:
                        multiple_preds[pred] += 1
                    else:
                        multiple_preds[pred] = 1
                preds.append(max(multiple_preds, key=multiple_preds.get))
            if not new_potential:
                preds.append(self.classifier.default_class)
            elif len(new_potential) == 1:
                preds.append(new_potential[0][1])
        return preds
    
class M1WithPrune(BaseEstimator):

    def __init__(self):
        self.minsup = 0.01
        self.minconf = 0.5
        self.classifier = None
        self.cba_rg_time=0
        self.m1_time=0

    def fit(self, X_train, y_train, sample_weight=None):
        dataset = X_train
        dataset = dataset.tolist()
        for i in range(len(y_train)):
            dataset[i].append(y_train[i])
        start_time = time.time()
        cars = rule_generator(dataset, self.minsup, self.minconf,True)
        #cars.prune_rules(training_dataset)
        cars.rules = cars.pruned_rules
        end_time = time.time()
        cba_rg_runtime = end_time - start_time

        start_time = time.time()
        classifier_m1 = M1(cars, dataset,True)
        end_time = time.time()
        cba_cb_runtime = end_time - start_time
        
        self.cba_rg_time=cba_rg_runtime+self.cba_rg_time
        self.m1_time=cba_cb_runtime+self.m1_time
        self.classifier = classifier_m1
    
    def predict(self, X_test):
        dataset = X_test
        preds = []
        for data in dataset:
            test_cond_set = []
            for column in range(0, len(data)-1):
                cond_set = [column, data[column]]
                test_cond_set.append(cond_set)
            potential = []
            for rule in self.classifier.rule_list:
                condset_list, classlabel = rule.get_condset()
                for condset in condset_list:
                    to_add = [condset_list, classlabel]
                    for i in test_cond_set:
                        if condset == i:
                            potential.append(to_add)
            idx_to_remove = []
            idx = 0
            for p in potential:
                for element in p[0]:
                    for i in test_cond_set:
                        if i[0] == element[0] and i[1] != element[1]:
                            idx_to_remove.append(idx)
                idx += 1
            all_idx = list(range(len(potential)))
            idx_to_keep = list(set(all_idx) - set(idx_to_remove))
            new_potential = []
            for i in idx_to_keep:
                new_potential.append(potential[i])
            if len(new_potential) > 1: # in case more than one potential matches, take majority
                multiple_preds = {}
                for p in new_potential:
                    pred = p[1]
                    if pred in multiple_preds:
                        multiple_preds[pred] += 1
                    else:
                        multiple_preds[pred] = 1
                preds.append(max(multiple_preds, key=multiple_preds.get))
            if not new_potential:
                preds.append(self.classifier.default_class)
            elif len(new_potential) == 1:
                preds.append(new_potential[0][1])
        return preds

if __name__ == "__main__":
    # using the relative path, all data sets are stored in datasets directory
    name_list=['australian','german','iris','tic-tac-toe','monks', 'messidor_features', 'seeds','zoo']
    for name in name_list:
        print(name)
        test_data_path = f'datasets/{name}.data'
        test_scheme_path = f'datasets/{name}.names'
        data, attributes, value_type = read(test_data_path, test_scheme_path)
        col_num = len(data[0])
        dataset = pre_process(data, attributes, value_type)
        training_dataset,test_dataset = train_test_split(dataset, test_size=0.3,random_state=42)
        
        X_train = [i[:-1] for i in training_dataset]
        X_test = [i[:-1] for i in test_dataset]
        y_train =[i[-1] for i in training_dataset]
        y_test = [i[-1] for i in test_dataset]

        # just choose one mode to experiment by removing one line comment and running
        start_time = time.time()
        bagging_classifier = BaggingClassifier(base_estimator=M1WithoutPrune(), n_estimators=10, random_state=42).fit(X_train, y_train)
        end_time = time.time()
        y_pred = bagging_classifier.predict(X_test)
        acc=0
        for i in range(len(y_pred)):
            if y_pred[i]==y_test[i]:
                acc = acc+1
        print('Bagging test without prune acc: ', acc/len(y_pred))
        print('time: ',end_time-start_time)
        print()
        start_time = time.time()
        bagging_classifier = BaggingClassifier(base_estimator=M1WithPrune(), n_estimators=10, random_state=42).fit(X_train, y_train)
        end_time = time.time()
        y_pred = bagging_classifier.predict(X_test)
        acc=0
        for i in range(len(y_pred)):
            if y_pred[i]==y_test[i]:
                acc = acc+1
        print('Bagging test with prune acc: ', acc/len(y_pred))
        print('time: ',end_time-start_time)
        print()

