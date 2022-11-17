from ruleitem import RuleItem
import sys

def cover(datacase, rule):
    for item in rule.cond_set:
        if datacase[item] != rule.cond_set[item]:
            return None #not covered by the rule
    if datacase[-1] == rule.class_label:
        return True #covered by the rule, and predicts correctly
    else:
        return False #covered by the rule, but it is wrong
    
def get_errors(r,dataset):
    e = 0
    for datacase in dataset:
        if cover(datacase, r) == False:
            e = e + 1
    return e

def prune(original_rule,min_error,pruned_rule,dataset):
        #print(pruned_rule)
        current_rule_cond_set = list(pruned_rule.cond_set)
        current_rule= pruned_rule
        
        if len(current_rule_cond_set) >= 2:
            #reduce attributes
            for attribute in current_rule.cond_set.keys():
                temp_cond_set = current_rule.cond_set.copy()
                temp_cond_set.pop(attribute)
                temp_rule = RuleItem(temp_cond_set, pruned_rule.class_label, dataset)
                temp_rule_error = get_errors(temp_rule,dataset)
                #if a subset of the attributes have lower errors than current attributes, return False
                if temp_rule_error <= min_error:
                    result = False
                    break
                #if a subset does not have lower errors, go deeper
                else:
                    result = prune(original_rule,min_error,temp_rule,dataset)
            return result

        else:
            #terminates when current cond_set <2
            #if they are not original rule, return false
            if original_rule.cond_set == pruned_rule.cond_set:
                return True
            else:
                return False
        
class FrequentRuleitems:
    '''A list of k-rule items'''
    def __init__(self,k):
        self.frequent_ruleitems_set = []
        self.k = k
        
    def size(self):
        return len(self.frequent_ruleitems_set)

    def add(self, rule_item):
        exist = False
        for item in self.frequent_ruleitems_set:
            if item.class_label == rule_item.class_label:
                if item.cond_set == rule_item.cond_set:
                    exist = True
                    break
                if len(rule_item.cond_set.keys())!=self.k:
                    break
        if exist == False:
            self.frequent_ruleitems_set.append(rule_item)

    def concat(self, sets):
        for item in sets.frequent_ruleitems_set:
            self.add(item)
            

class Car:
    def __init__(self):
        self.rules = []
        self.pruned_rules = []

    def add_rule_item(self, rule_item, minconf):
        if rule_item.confidence >= minconf:
            if rule_item in self.rules:
                return
            for item in self.rules:
                if item.cond_set == rule_item.cond_set and item.confidence < rule_item.confidence:
                    self.rules.remove(item)
                    self.rules.add(rule_item)
                    return
                elif item.cond_set == rule_item.cond_set and item.confidence >= rule_item.confidence:
                    return
            self.rules.append(rule_item)

    def generate_rules(self, frequent_ruleitems, minsup, minconf):
        for item in frequent_ruleitems.frequent_ruleitems_set:
            self.add_rule_item(item, minconf)

    def prune_rules(self, dataset):
        for rule in self.rules:
            pruned_rule_value = prune(rule,get_errors(rule,dataset), rule, dataset)
            if pruned_rule_value == True:
            #print(pruned_rule.cond_set)
                is_existed = False
                for r in self.pruned_rules:
                    if r.class_label == rule.class_label:
                        if r.cond_set == rule.cond_set:
                            is_existed = True
                            break

                if is_existed==False:
                    self.pruned_rules.append(rule)

    def concat(self, car, minsup, minconf):
        for item in car.rules:
            self.add_rule_item(item, minconf)
        for item in car.pruned_rules:
            is_existed = False
            for rule in self.pruned_rules:
                if rule.class_label == item.class_label:
                    if rule.cond_set == item.cond_set:
                        is_existed = True
                        break
            if is_existed == False:
                self.pruned_rules.append(item)
            
    def print_rules(self,is_prune):
        if is_prune:
            for i in self.pruned_rules:
                print(i.cond_set)
        else:
            for i in self.rules:
                print(i.cond_set)




def merge(item1, item2, dataset):
    if item1.class_label != item2.class_label:
        return None
    category1 = set(item1.cond_set)
    category2 = set(item2.cond_set)
    c1_list = sorted(category1)
    c2_list = sorted(category2)

    if category1 == category2:
        return None
    for i in range(len(c1_list)):
        if c1_list[i] != c2_list[i]:
            if i != len(category1)-1:
                return None

    intersect = category1 & category2
    for item in intersect:
        if item1.cond_set[item] != item2.cond_set[item]:
            return None

    union = category1 | category2
    new_cond_set = dict()
    for item in union:
        if item in category1:
            new_cond_set[item] = item1.cond_set[item]
        else:
            new_cond_set[item] = item2.cond_set[item]
    new_ruleitem = RuleItem(new_cond_set, item1.class_label, dataset)
    return new_ruleitem


def generate_candidate(frequent_ruleitems, dataset):
    k=frequent_ruleitems.k
    frequent_ruleitems_copy = FrequentRuleitems(k)
    result = FrequentRuleitems(k+1)
    frequent_ruleitems_copy.concat(frequent_ruleitems)
    for item1 in frequent_ruleitems.frequent_ruleitems_set:
        for item2 in frequent_ruleitems_copy.frequent_ruleitems_set:
            new_ruleitem = merge(item1, item2, dataset)
            if new_ruleitem is not None:
                result.add(new_ruleitem)
                if result.size() >= 1000:
                    return result
    return result


def rule_generator(dataset, minsup, minconf,do_prune):
    k=1
    frequent_ruleitems = FrequentRuleitems(1)
    car = Car()
    class_label = sorted(set([i[-1] for i in dataset])) #get unique labels
    
    # this generates frequent 1-ruleitem
    for column in range(0, len(dataset[0])-1):
        distinct_value = sorted(set([i[column] for i in dataset]))
        for value in distinct_value:
            cond_set = {column: value}
            for classes in class_label:
                rule_item = RuleItem(cond_set, classes, dataset)
                if rule_item.support >= minsup:
                    frequent_ruleitems.add(rule_item)
                    
    
    car.generate_rules(frequent_ruleitems, minsup, minconf)
    
    #optionally prune rules
    if do_prune:
        car.prune_rules(dataset)
        current_cars_number = len(car.pruned_rules)
    else:
        current_cars_number = len(car.rules)
    
    all_car=car
    #if do_prune, limit the rule to be <6, because pruning takes far too long - O(2^k) for each call of prune()!!! 
    while frequent_ruleitems.size() > 0 and current_cars_number <= 2000 and (k<6 or (not do_prune)):
        #print(k)
        k=k+1
        candidate = generate_candidate(frequent_ruleitems, dataset)
        frequent_ruleitems = FrequentRuleitems(k)
        car = Car()
        for item in candidate.frequent_ruleitems_set:
            if item.support >= minsup:
                frequent_ruleitems.add(item)
        car.generate_rules(frequent_ruleitems, minsup, minconf)
        if do_prune:
            car.prune_rules(dataset)
        all_car.concat(car, minsup, minconf)
    #print("done")
    if do_prune:
        all_car.rules=all_car.pruned_rules
        all_car.pruned_rules=[]
        all_car.prune_rules(dataset)
    # print(len(all_car.rules))
    # print(len(all_car.pruned_rules))
    return all_car
