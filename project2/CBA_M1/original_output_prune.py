from read import read
from pre_processing import pre_process
from cba_rg import rule_generator
from cba_cb_m1 import M1
from cba_cb_m1 import cover
import time
from sklearn.model_selection import train_test_split

def get_acc(classifier, dataset):
    incorrect = 0
    cover_value = False
    for datacase in dataset:
        for rule in classifier.rule_list:
            cover_value = cover(datacase, rule)
            if cover_value == True:
                break
        if cover_value == False:
            if classifier.default_class != datacase[-1]:
                incorrect = incorrect+1
    return 1-incorrect/len(dataset)

def m1_with_prune(data_path, scheme_path, minsup=0.01, minconf=0.5):
    data, attributes, value_type = read(data_path, scheme_path)
    dataset = pre_process(data, attributes, value_type)
    training_dataset,test_dataset = train_test_split(dataset, test_size=0.3,random_state=42)

    start_time = time.time()
    cars = rule_generator(training_dataset, minsup, minconf,True)
    end_time = time.time()
    car_runtime = end_time - start_time
    
    start_time = time.time()
    classifier_m1 = M1(cars, training_dataset, True)
    end_time = time.time()
    m1_runtime = end_time - start_time

    acc = get_acc(classifier_m1, test_dataset)

    print(f"Accuracy with pruning: {acc}")
    print(f"Car's run time with pruning: {car_runtime}")
    print(f"M1's run time with pruning: {m1_runtime}")


if __name__ == "__main__":
    name_list=['australian','german','iris','tic-tac-toe','monks', 'messidor_features', 'seeds','zoo']
    for name in name_list:
        print(name)
        test_data_path = f'datasets/{name}.data'
        test_scheme_path = f'datasets/{name}.names'
        m1_with_prune(test_data_path, test_scheme_path)
        print()