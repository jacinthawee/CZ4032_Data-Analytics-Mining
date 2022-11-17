from functools import cmp_to_key
import sys

def cover(datacase, rule):
    for item in rule.cond_set:
        if datacase[item] != rule.cond_set[item]:
            return None #not covered by the rule
    if datacase[-1] == rule.class_label:
        return True #covered by the rule, and predicts correctly
    else:
        return False #covered by the rule, but it is wrong


class Classifier:
    def __init__(self):
        self.rule_list = list()
        self.default_class = None
        self.error_list = list()
        self.default_class_list = list()

    # add rules
    def insert(self, rule, dataset):
        self.rule_list.append(rule)           
        self.select_default_class(dataset)     
        self.compute_error(dataset)            

    # select the majority class in the remaining data
    def select_default_class(self, dataset):
        class_column = [x[-1] for x in dataset]
        class_label = set(class_column)
        max = 0
        current_default_class = None
        for label in class_label:
            if class_column.count(label) >= max:
                max = class_column.count(label)
                current_default_class = label
        self.default_class_list.append(current_default_class)

    # compute the sum of errors
    def compute_error(self, dataset):
        if len(dataset) <= 0:
            self.error_list.append(sys.maxsize)
            return

        error = 0
        # the number of errors by the rules
        for case in dataset:
            is_cover_value = False
            for rule in self.rule_list:
                if cover(case, rule):
                    is_cover_value = True
                    break
            if not is_cover_value:
                error = error + 1

        # the number of errors by the default class in the training set
        class_column = [x[-1] for x in dataset]
        error = len(class_column) - class_column.count(self.default_class_list[-1])+error
        self.error_list.append(error)

    # remove useless rules
    def remove(self):
        index = self.error_list.index(min(self.error_list))
        self.rule_list = self.rule_list[:(index+1)]
        self.error_list = None
        self.default_class = self.default_class_list[index]
        self.default_class_list = None




def sort(rule_list):
    def cmp_method(a, b):
        if a.confidence < b.confidence:     
            return 1
        elif a.confidence == b.confidence:
            if a.support < b.support:       
                return 1
            elif a.support == b.support:
                if len(a.cond_set) < len(b.cond_set):   
                    return -1
                elif len(a.cond_set) == len(b.cond_set):
                    return 0
                else:
                    return 1
            else:
                return -1
        else:
            return -1
    rule_list.sort(key=cmp_to_key(cmp_method))
    return rule_list

def M1(cars, dataset, do_prune):
    classifier = Classifier()
    if do_prune==False:
        cars_list = sort(cars.rules)
    else:
        cars_list = sort(cars.pruned_rules)
    for rule in cars_list:
        temp = []
        mark = False
        for i in range(len(dataset)):
            is_cover_value = cover(dataset[i], rule)
            if is_cover_value is not None:
                temp.append(i)
                if is_cover_value:
                    mark = True
        if mark:
            temp_dataset = list(dataset)
            for index in temp:
                temp_dataset[index] = []
            while [] in temp_dataset:
                temp_dataset.remove([])
            dataset = temp_dataset
            classifier.insert(rule, dataset)
    classifier.remove()
    return classifier