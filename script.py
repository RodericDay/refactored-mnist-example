import numpy as np


def create_model():
    '''
    input_shape needs to match source information
    '''
    import os
    os.environ["KERAS_BACKEND"] = "theano"
    from keras import backend as K
    K.set_image_dim_ordering('th')

    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation, Flatten
    from keras.layers import Conv2D, MaxPooling2D

    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(1,28,28)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def train(data, model):
    X = np.array([[a] for a in data["x_train"]])/255
    y = data["y_train"]
    Y = np.vstack([y==u for u in np.unique(y)]).T
    model.load_weights("weights.h5")
    model.fit(X, Y, batch_size=32, epochs=1, verbose=True)
    model.save_weights("weights.h5", overwrite=True)


def test(data, model, N):
    X = np.array([[a] for a in data["x_test"]])/255
    rns = np.random.choice(len(X), N, replace=False)
    yield X[rns]

    model.load_weights("weights.h5")
    ps = model.predict(X[rns]).argmax(axis=1)
    yield ps

    yield data["y_test"][rns]


def visualize(Xs, xs, ys):
    from matplotlib import pyplot as plt
    N = np.ceil(np.sqrt(ys.size)).astype(int)
    fig, axes = plt.subplots(N, N)
    for ax in axes.flat:
        ax.set_axis_off()
    for i, (ax, X, x, y) in enumerate(zip(axes.flat, Xs, xs, ys)):
        ax.matshow(X[0], cmap="gray_r")
        if x==y:
            ax.text(0, 0, x)
        else:
            ax.text(0, 0, f"{x} ({y})", color="red")
    plt.savefig("example.png")


def main():
    data = np.load("mnist.npz")
    model = create_model()
    # train(data, model)
    Xs, xs, ys = test(data, model, 64)
    visualize(Xs, xs, ys)

if __name__ == '__main__':
    main()
