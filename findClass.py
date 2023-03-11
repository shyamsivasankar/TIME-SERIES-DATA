import pickle
with open('rfc_tsm.pkl', 'rb') as f:
    rfc= pickle.load(f)
    pred = rfc.predict(data.iloc[-1][:-1].values.reshape(1, -1))
    # print(pred)
    # print(model)
    keys = []
    for key, value in model.items():
        if value == pred:
            print(key)
            break