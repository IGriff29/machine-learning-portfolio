from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

def main():
    X,y=load_breast_cancer(return_X_y=True)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    scaler=StandardScaler(); X_train=scaler.fit_transform(X_train); X_test=scaler.transform(X_test)
    model=keras.Sequential([keras.layers.Input(shape=(X_train.shape[1],)), keras.layers.Dense(32,activation="relu"), keras.layers.Dropout(.2), keras.layers.Dense(16,activation="relu"), keras.layers.Dense(1,activation="sigmoid")])
    model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
    stop=keras.callbacks.EarlyStopping(patience=5,restore_best_weights=True)
    model.fit(X_train,y_train,validation_split=.2,epochs=100,batch_size=32,callbacks=[stop],verbose=0)
    loss,acc=model.evaluate(X_test,y_test,verbose=0)
    print(f"Test accuracy: {acc:.3f}")

if __name__ == "__main__": main()
