class RuleItem:
    def __init__(self, cond_set, class_label, dataset):
        self.cond_set = cond_set # {index: value}
        self.class_label = class_label
        self.cond_support_count, self.rule_support_count = self.support_count(dataset)
        self.support = self.rule_support_count/len(dataset)
        if self.cond_support_count != 0:
            self.confidence = self.rule_support_count / self.cond_support_count
        else:
            self.confidence = 0

    def support_count(self, dataset):
        '''calculate the support count for rule and condset over the whole dataset'''
        cond_support_count = 0
        rule_support_count = 0
        for case in dataset:
            contained = True
            for index in self.cond_set:
                if self.cond_set[index] != case[index]:
                    contained = False
                    break
            if contained==True:
                cond_support_count = cond_support_count+1
                if self.class_label == case[-1]:
                    rule_support_count = rule_support_count+1
        return cond_support_count, rule_support_count

    def get_condset(self):
        condset_list = []
        for item in self.cond_set:
            condset = [item, self.cond_set[item]]
            condset_list.append(condset)
        classlabel = self.class_label
        return condset_list, classlabel