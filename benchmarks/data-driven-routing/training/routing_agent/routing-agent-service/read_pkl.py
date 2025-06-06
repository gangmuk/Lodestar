import pickle

# Reading a pickle file
with open('final_model-working-model/history.pkl', 'rb') as file:
    data = pickle.load(file)

print(data)