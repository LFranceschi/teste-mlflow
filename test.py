import mlflow
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("teste_mlflow_nn")

# Carregando os dados
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Configuração do modelo
hidden_layer_sizes = (3, 3)
max_iter = 5000
alpha = 0.0001

# Iniciando um run
with mlflow.start_run():
    
    # Logando os parâmetros
    mlflow.log_param("hidden_layer_sizes", hidden_layer_sizes)
    mlflow.log_param("max_iter", max_iter)
    mlflow.log_param("alpha", alpha)
    
    # Treinando o modelo
    rf = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, alpha=alpha)
    rf.fit(X_train, y_train)
    
    # Logando a curva de loss
    for step, loss in enumerate(rf.loss_curve_):
        mlflow.log_metric("loss", loss, step=step)
    
    # testar em um conjunto de teste e logar a métrica
    mlflow.log_metric("mse", rf.score(X_test, y_test))
    
    # Fazendo predições
    predictions = rf.predict(X_test)
    
    print(predictions)
    
# O run é encerrado automaticamente quando saímos do bloco 'with'
