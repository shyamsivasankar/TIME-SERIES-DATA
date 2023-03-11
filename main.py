from fastapi import FastAPI,File
import findClass as fc
from Visualization import visualize

app = FastAPI()



@app.post("/predict")
async def predict(data_set: bytes = File()):
    df,model = fc.classifier(data_set)
    # df = pd.read_csv(BytesIO(data_set))
    visualize(df,model) 
    return {"file_size": len(data_set),"model":model}