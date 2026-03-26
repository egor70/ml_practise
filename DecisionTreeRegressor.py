import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor  # Используем регрессор
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree

# Загрузка данных
data = pd.read_csv(r'C:\Users\User\Downloads\Admission_Predict.csv')

# Предварительная обработка
X = data.drop(['Serial No.', 'Chance of Admit'], axis=1)
y = data['Chance of Admit']

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели
clf = DecisionTreeRegressor(max_depth=4, random_state=42)  # Используем регрессор
clf.fit(X_train, y_train)

# Прогноз
y_pred = clf.predict(X_test)

# Оценка качества модели
print("Средняя абсолютная ошибка:", metrics.mean_absolute_error(y_test, y_pred))
print("Среднеквадратичная ошибка:", metrics.mean_squared_error(y_test, y_pred))
print("Коэффициент детерминации (R²):", metrics.r2_score(y_test, y_pred))

# Визуализация дерева
plt.figure(figsize=(20,10))
tree.plot_tree(clf,
              filled=True,
              feature_names=X.columns,
              rounded=True,
              fontsize=10)
plt.show()
