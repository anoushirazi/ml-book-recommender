import pickle

# Load the model from file
with open("svd_model.pkl", "rb") as f:
    model = pickle.load(f)

print(type(model))   # See what kind of object it is
print(model)         # Prints details 
