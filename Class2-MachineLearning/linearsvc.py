from sklearn.svm import LinearSVC

comp1 = [1, 1, 1]
comp2 = [0, 0, 0]
comp3 = [1, 0, 1]
comp4 = [0, 1, 0]
comp5 = [1, 1, 0]
comp6 = [0, 0, 1]

train_data = [comp1, comp2, comp3, comp4, comp5, comp6]
train_labels = ['S', 'N', 'S', 'N', 'S', 'S']

model = LinearSVC()
model.fit(train_data, train_labels)

test1 = [1, 0, 0]
test2 = [0, 1, 1]
test3 = [1, 0, 1]

test_data = [test1, test2, test3]
predicted = model.predict(test_data)

map_prection = {'S': 'Solúvel', 'N': 'Não Solúvel'}

for i, prediction in enumerate(predicted):
    print(f"O composto {i} pode ser considerado {map_prection[prediction]}")