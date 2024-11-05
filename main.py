import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Carregar o conjunto de dados
df = pd.read_csv('gutenberg_book_deer.csv')

# Configurar dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo e o tokenizer do DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)

# Criar uma nova coluna combinando título e assunto
df['TitleSubject'] = df['Title'].fillna('') + ' ' + df['Subject'].fillna('')

# Função para gerar embeddings
def get_embeddings(texts, batch_size=32, num_rows=None):
    if num_rows is not None:
        texts = texts[:num_rows]
    
    embeddings_list = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(batch.tolist(), return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        embeddings_list.append(batch_embeddings)
        print(f"Processando lote {i // batch_size + 1} de {len(texts) // batch_size + 1}")
        
    return np.vstack(embeddings_list)

# Definir o número de linhas a serem utilizadas
num_rows = 500 
embeddings = get_embeddings(df['TitleSubject'], batch_size=32, num_rows=num_rows)
print(f"Shape of embeddings: {embeddings.shape}")

# Etapa 1: Métrica de avaliação e comparação
def evaluate_embeddings_classification(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, test_size=0.2, random_state=42)
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Classificação de Acurácia: {accuracy:.4f}")
    return accuracy

# Definir um problema de classificação proxy usando os assuntos dos livros
labels = df['Subject'].astype('category').cat.codes[:num_rows]
accuracy_distilbert = evaluate_embeddings_classification(embeddings, labels)

# Comparação com outros modelos (neste caso, embeddings aleatórios como baseline)
random_embeddings = np.random.rand(num_rows, 768)
accuracy_random = evaluate_embeddings_classification(random_embeddings, labels)

# Visualização das métricas
metrics = {'DistilBERT Embeddings': accuracy_distilbert, 'Random Embeddings': accuracy_random}
plt.bar(metrics.keys(), metrics.values())
plt.xlabel('Modelo de Embeddings')
plt.ylabel('Acurácia')
plt.title('Comparação de Acurácia entre Embeddings')
plt.show()

# Definir a arquitetura da rede neural para ajuste fino
class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Inicializar a rede, função de perda e otimizador
input_dim = 768  # Dimensão dos embeddings
output_dim = 768  # Dimensão de saída
model_nn = SimpleNN(input_dim, output_dim).to(device)
criterion = nn.MSELoss()  # Função de perda
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

# Processo de treinamento
num_epochs = 10  # Ajuste conforme necessário
for epoch in range(num_epochs):
    model_nn.train()
    optimizer.zero_grad()
    embeddings_tensor = torch.tensor(embeddings).float().to(device)
    outputs = model_nn(embeddings_tensor)
    loss = criterion(outputs, embeddings_tensor)
    loss.backward()
    optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Função para desenhar a topologia do DistilBERT
def plot_distilbert_architecture():
    plt.figure(figsize=(10, 6))
    layers = ['Input Tokens', 'Token Embeddings', 'Position Embeddings', 'Transformer Block (6)', 'Output Embeddings']
    heights = np.arange(len(layers))
    plt.barh(heights, np.random.randint(1, 10, size=len(layers)), color='skyblue')

    for i, layer in enumerate(layers):
        plt.text(0.5, i, layer, ha='center', va='center', fontsize=12, color='black')

    plt.yticks(heights, [])
    plt.title('Topologia da Rede Neural DistilBERT', fontsize=16)
    plt.xlabel('Camadas', fontsize=14)
    plt.grid(axis='x', linestyle='--')

    # Salvar a figura
    plt.savefig('distilbert_architecture.png', bbox_inches='tight')
    plt.show()

# Chamar a função para plotar a arquitetura
plot_distilbert_architecture()

# Visualização dos embeddings originais usando t-SNE
def visualize_embeddings(embeddings, title):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

visualize_embeddings(embeddings, "Visualização dos Embeddings Originais")

# Visualização dos embeddings ajustados usando t-SNE
adjusted_embeddings = model_nn(torch.tensor(embeddings).float().to(device)).cpu().detach().numpy()
visualize_embeddings(adjusted_embeddings, "Visualização dos Embeddings Ajustados")

# Função para recomendar livros
def recomendar_livros(titulo, embeddings, df, top_n=5):
    idx = df[df['Title'].str.contains(titulo, case=False, na=False)].index[0]
    titulo_embedding = embeddings[idx].reshape(1, -1)
    similaridades = cosine_similarity(titulo_embedding, embeddings).flatten()
    indices_recomendados = similaridades.argsort()[-top_n-1:-1][::-1]
    recomendacoes = df.iloc[indices_recomendados][['Title', 'Subject', 'URL']]
    return recomendacoes

# Sistema de busca
def buscar_livros(query, embeddings, df, top_n=10):
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    similaridades = cosine_similarity(query_embedding, embeddings).flatten()
    indices_recomendados = similaridades.argsort()[-top_n-1:-1][::-1]
    return df.iloc[indices_recomendados][['Title', 'Subject', 'URL']].assign(Similarity_Score=similaridades[indices_recomendados])

# Etapa 3: Experimento com frases de mockup
mock_query = "blablabla nonsensical"
inputs = tokenizer(mock_query, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
with torch.no_grad():
    mock_query_embedding = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
similaridades_mock = cosine_similarity(mock_query_embedding, embeddings).flatten()
indices_mock = similaridades_mock.argsort()[-5:][::-1]

# Exibir os resultados do experimento
print("\n--- Resultados do Experimento com Frase de Mockup ---")
for i, idx in enumerate(indices_mock):
    print(f"Recomendação {i + 1}:")
    print(f"Título: {df.iloc[idx]['Title']}")
    print(f"Similaridade: {similaridades_mock[idx]:.4f}")
    print("---")

# Testes do sistema de busca
print("\n--- Teste 1: Consulta que produz 10 resultados ---")
query1 = "Science"
resultados1 = buscar_livros(query1, embeddings, df, top_n=10)
print(resultados1)

print("\n--- Teste 2: Consulta que produz menos de 10 resultados ---")
query2 = "Stingy Jack"
resultados2 = buscar_livros(query2, embeddings, df, top_n=10)
print(resultados2)

print("\n--- Teste 3: Consulta que produz algo não óbvio ---")
query3 = "Haphazard"
resultados3 = buscar_livros(query3, embeddings, df, top_n=10)
print(resultados3)
