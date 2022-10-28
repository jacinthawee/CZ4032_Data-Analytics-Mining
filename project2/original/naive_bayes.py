import csv
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# Read dataset and convert into a list.
# path: directory of *.data file.
def read_data(path):
    data = []
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        for line in reader:
            data.append(line)
        while [] in data:
            data.remove([])
    return data


# Read scheme file *.names and write down attributes and value types.
# path: directory of *.names file.
def read_scheme(path):
    with open(path, 'r') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        attributes = next(reader)
        value_type = next(reader)
    return attributes, value_type


# convert string-type value into float-type.
# data: data list returned by read_data.
# value_type: list returned by read_scheme.
def str2numerical(data, value_type):
    size = len(data)
    columns = len(data[0])
    for i in range(size):
        for j in range(columns-1):
            if value_type[j] == 'numerical' and data[i][j] != '?':
                data[i][j] = float(data[i][j])
    return data


# Main method in this file, to get data list after processing and scheme list.
# data_path: tell where *.data file stores.
# scheme_path: tell where *.names file stores.
def read(data_path, scheme_path):
    data = read_data(data_path)
    attributes, value_type = read_scheme(scheme_path)
    data = str2numerical(data, value_type)
    return data, attributes, value_type


# just for test
if __name__ == '__main__':
    import pre_processing
    name_list=['australian','german','iris','tic-tac-toe','monks', 'messidor_features', 'seeds']
    gnb = GaussianNB()
    for name in name_list:
        print(f"Dataset: {name}")
        test_data_path = f'datasets/{name}.data'
        test_scheme_path = f'datasets/{name}.names'
        test_data, test_attributes, test_value_type = read(test_data_path, test_scheme_path)
        result_data = pre_processing.pre_process(test_data, test_attributes, test_value_type)
        x=[]
        y=[]
        for i in result_data:
            x.append(i[:-1])
            y.append(i[-1])
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train).predict(X_test)
        mis_labelled=(y_test != y_pred).sum()
        print("accuracy: ",1-mis_labelled/len(X_test))
    
    name='zoo'
    print(f"Dataset: {name}")
    test_data_path = f'datasets/{name}.data'
    test_scheme_path = f'datasets/{name}.names'
    test_data, test_attributes, test_value_type = read(test_data_path, test_scheme_path)
    result_data = pre_processing.pre_process(test_data, test_attributes, test_value_type)
    x=[]
    y=[]
    for i in result_data:
        last=i[:-1][-1]
        last=int(last)
        process=i[:-2]
        process.append(last)
        x.append(process)
        y.append(i[-1])
    #print(x)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train, y_train).predict(X_test)
    mis_labelled=(y_test != y_pred).sum()
    print("error rate: ",mis_labelled/len(X_test))
