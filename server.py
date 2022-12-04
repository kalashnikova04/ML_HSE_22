import joblib
import uvicorn
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List
from train import find_num, find_torque, find_max_torque
import pandas as pd
import shutil

app = FastAPI()
imp = joblib.load("./imp.joblib")
model = joblib.load("./poly_model.joblib")
poly = joblib.load("./poly_features.joblib")
scale = joblib.load("./scale_features_mm.joblib")


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str 
    engine: str
    max_power: str
    torque: str
    seats: float

class Items(BaseModel):
    objects: List[Item]

@app.post("/predict_item")
async def predict_item(item: Item):

    features = [[
        item.name,
        item.year,
        
        item.km_driven,
        item.fuel,
        item.seller_type,
        item.transmission,
        item.owner,
        item.mileage,
        item.engine,
        item.max_power,
        item.torque,
        item.seats
    ]]

    df_test = pd.DataFrame(features, columns=['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'max_power', 'torque', 'seats'])
    df_test['mileage'] = df_test['mileage'].apply(find_num)
    df_test['engine'] = df_test['engine'].apply(find_num)
    df_test['max_power'] = df_test['max_power'].apply(find_num)
    df_test['nt'] = df_test['torque'].apply(find_torque)
    df_test['max_torque'] = df_test['torque'].apply(find_max_torque)
    df_test['torque'] = df_test['nt']
    df_test.drop('nt', inplace=True, axis=1)

    df_test_mis = df_test.loc[:, 'mileage':'max_torque']
    df_test_no_mis = df_test.loc[:, 'name':'owner']
    df_test_mis_upd = imp.transform(df_test_mis)
    df_test_mis = pd.DataFrame(data=df_test_mis_upd, columns=df_test_mis.columns)
    df_test_upd = pd.concat([df_test_no_mis, df_test_mis], axis=1)
    df_test_upd['engine'] = df_test_upd['engine'].astype('int64')
    df_test_upd['seats'] = df_test_upd['seats'].astype('int64')
    df_test_upd['max_torque'] = df_test_upd['max_torque'].astype('int64')
 
    num_features_mask = (df_test_upd.dtypes == 'int64').values | (df_test_upd.dtypes == 'float64').values

    X_test = df_test_upd[df_test_upd.columns[num_features_mask]]

    features_test = scale.transform(X_test)
    X_test_mm = pd.DataFrame(data=features_test, columns=X_test.columns)

    X_test_poly1 = poly.transform(X_test_mm)


    prediction = model.predict(X_test_poly1)[0]
    return  prediction
    




@app.post("/predict_items")
async def predict_items(file: UploadFile = File(...)):

    with open(f'{file.filename}', 'wb') as data:
        shutil.copyfileobj(file.file, data)

        f = file.filename

    df_test = pd.read_csv(f)
    df_test['mileage'] = df_test['mileage'].apply(find_num)
    df_test['engine'] = df_test['engine'].apply(find_num)
    df_test['max_power'] = df_test['max_power'].apply(find_num)
    df_test['nt'] = df_test['torque'].apply(find_torque)
    df_test['max_torque'] = df_test['torque'].apply(find_max_torque)
    df_test['torque'] = df_test['nt']
    df_test.drop(['nt', 'selling_price'], inplace=True, axis=1)

    df_test_mis = df_test.loc[:, 'mileage':'max_torque']
    df_test_no_mis = df_test.loc[:, 'name':'owner']
    df_test_mis_imp = imp.transform(df_test_mis)
    df_test_mis_upd = pd.DataFrame(data=df_test_mis_imp, columns=df_test_mis.columns)
    df_test_upd = pd.concat([df_test_no_mis, df_test_mis_upd], axis=1)
    
    df_test_upd['engine'] = df_test_upd['engine'].astype('int64')
    df_test_upd['seats'] = df_test_upd['seats'].astype('int64')
    df_test_upd['max_torque'] = df_test_upd['max_torque'].astype('int64')
    num_features_mask = (df_test_upd.dtypes == 'int64').values | (df_test_upd.dtypes == 'float64').values

    X_test = df_test_upd[df_test_upd.columns[num_features_mask]]

    features_test = scale.transform(X_test)
    X_test_mm = pd.DataFrame(data=features_test, columns=X_test.columns)

    X_test_poly1 = poly.transform(X_test_mm)

    prediction = model.predict(X_test_poly1).tolist()
    print(prediction)
    return  prediction
    



if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=80)