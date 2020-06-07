import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels.api as sm
def lda(X,Y,Xtest_data,v,q,w):
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_train = sc.fit_transform(X)
    X_test = sc.transform(Xtest_data)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    lda = LinearDiscriminantAnalysis(n_components=2)
    X_train = lda.fit_transform(X_train, Y)
    X_test = lda.transform(X_test)

    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(random_state=0)
    classifier.fit(X_train, Y)

    y_pred = classifier.predict(X_test)
    print(X_test)

    for j in range(len(Y)):
        if Y[j] == 1:
            plt.scatter(q[j], w[j], color='b', marker='o', s=30)

    for j in range(len(Y)):
        if Y[j] == 2:
            plt.scatter(q[j], w[j], color='r', marker='o', s=30)

    plt.title('Linear Discriminant Analysis Based On Rain And Moisture')
    plt.xlabel('rain')
    plt.ylabel('moisture')
    plt.show()

    for j in range(len(Y)):
        if Y[j] == 1:
            plt.scatter(v[j], w[j], color='b', marker='o', s=30)

    for j in range(len(Y)):
        if Y[j] == 2:
            plt.scatter(v[j], w[j], color='r', marker='o', s=30)

    plt.title('Linear Discriminant Analysis Based On Temperature And Moisture')
    plt.xlabel('temperature')
    plt.ylabel('moisture')
    plt.show()

    for j in range(len(Y)):
        if Y[j] == 1:
            plt.scatter(v[j], q[j], color='b', marker='o', s=10)

    for j in range(len(Y)):
        if Y[j] == 2:
            plt.scatter(v[j], q[j], color='r', marker='o', s=10)

    plt.title('Linear Discriminant Analysis Based On temperature And Rain')
    plt.xlabel('temperature')
    plt.ylabel('rain')
    plt.show()

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(len(Y)):
        if Y[j] == 1:
            ax.scatter(v[j], q[j],w[j], color='b', marker='o', s=10)

    for j in range(len(Y)):
        if Y[j] == 2:
            ax.scatter(v[j], q[j],w[j], color='r', marker='o', s=10)

    ax.set_xlabel('Temperature')
    ax.set_ylabel('Rainfall')
    ax.set_zlabel('Moisture')

    plt.show()

    return y_pred

def estimate_coef(x, y):
    n = np.size(x)

    m_x, m_y = np.mean(x), np.mean(y)

    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


def plot_regression_line(x, y, b):
    plt.scatter(x, y, color="m",
                marker="o", s=10)

    y_pred = b[0] + b[1] * x

    plt.plot(x, y_pred, color="g")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


def ols(X,Z,temp,rain,moisture):
    df = pd.read_excel("C:\\Users\\Suvir\\OpenLab\\Dataset.xlsx")
    X = sm.add_constant(X)
    model = sm.OLS(Z, X).fit()
    print(model.summary())
    print(model.params)

    y = model.params[0] + temp * model.params[1] + rain * model.params[2] + moisture * model.params[3]
    return y

def main():
    df = pd.read_excel("C:\\Users\\Suvir\\OpenLab\\Dataset.xlsx")
    X = df.iloc[:, 1:4].values
    Y = df.iloc[:, 0].values
    Z= df.iloc[:, 4].values
    v = df.iloc[:, 1].values
    q = df.iloc[:, 2].values
    w = df.iloc[:, 3].values
    temp = float(input("Enter Temperature"))
    rain = float(input("Enter Rainfall"))
    new_df = df[['Temperature', 'Rainfall', 'Moisture']]
    x = np.array([new_df["Rainfall"]])
    y = np.array(new_df["Moisture"])

    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(x, y, b)
    y_pred = b[0] + b[1] * rain
    print("Approx Soil Moisture is -->")
    print(y_pred)
    # lda
    test_data = [[temp, rain, y_pred]]
    print(test_data)
    z = lda(X,Y,test_data,v,q,w)
    print(z)
    print("If\n1--Wheat is a better option to cultivate in this condition\n2--Maize is a better option to cultivate in this condition")

    z=ols(X,Z,temp,rain,y_pred)
    print("Approx Yield is -->")
    print(z)

    x = np.array([df["Yield"]])
    y = np.array(df["Cost of Cultivation"])

    b = estimate_coef(x, y)
    print("Estimated coefficients:\nb_0 = {}  \nb_1 = {}".format(b[0], b[1]))
    plot_regression_line(x, y, b)
    y_pred = b[0] + b[1] * z
    print("Approx Cost of Cultivation is -->")
    print(y_pred)


if __name__ == "__main__":
    main()

