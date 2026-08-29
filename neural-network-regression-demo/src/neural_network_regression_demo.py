import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

def main():
    X,y=load_diabetes(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42)
    scaler=StandardScaler(); X_train=scaler.fit_transform(X_train); X_test=scaler.transform(X_test)
    model=keras.Sequential([keras.layers.Input(shape=(X_train.shape[1],)), keras.layers.Dense(64,activation="relu"), keras.layers.Dense(32,activation="relu"), keras.layers.Dense(1)])
    model.compile(optimizer="adam",loss="mse")
    stop=keras.callbacks.EarlyStopping(patience=8,restore_best_weights=True)
    model.fit(X_train,y_train,validation_split=.2,epochs=150,batch_size=32,callbacks=[stop],verbose=0)
    pred=model.predict(X_test,verbose=0).ravel()
    print("RMSE:", round(mean_squared_error(y_test,pred)**0.5,3))

if __name__ == "__main__": main()
