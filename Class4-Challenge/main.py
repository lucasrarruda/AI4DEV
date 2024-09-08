from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Dados de exemplo
textos = [
    # Tecnologia
    "A Apple lançou um novo iPhone com características avançadas",
    "Tecnologia 5G está se espalhando rapidamente pelo mundo",
    "Apple está inovando com novos produtos para o mercado",
    "A revolução tecnológica está transformando as empresas",
    "Tecnologia wearable está mudando o mercado de gadgets",
    "Avanços tecnológicos em dispositivos móveis são impressionantes",
    "Novidades sobre o acordo de paz internacional",
    "Os últimos lançamentos de tecnologia estão disponíveis para compra",
    "A nova tecnologia de IA promete revolucionar o setor de saúde",
    "As inovações em computação quântica estão progredindo rapidamente",
    "O mercado de tecnologia está em constante evolução",
    "Novos dispositivos de realidade aumentada estão sendo lançados",
    "Tecnologia de blockchain está sendo adotada por mais empresas",

    # Esportes
    "O campeonato de basquete está no meio da temporada",
    "Os jogos de futebol estão atraindo muitos espectadores",
    "O time de futebol venceu a partida final do campeonato",
    "Fórum sobre os últimos resultados de futebol europeu",
    "O torneio de futebol regional atraiu muitos fãs",
    "Competição de esportes eletrônicos ganhou popularidade",
    "O jogo de vôlei entre Brasil e Argentina foi emocionante",
    "A final do campeonato de tênis será realizada no próximo mês",
    "O time de natação bateu recordes nacionais no campeonato",
    "O evento de esportes radicais atraiu uma grande multidão",
    "O novo time de rugby teve um desempenho impressionante",
    "A maratona anual trouxe corredores de todo o mundo",
    "O campeonato mundial de esportes eletrônicos começou ontem",

    # Política
    "A nova eleição será decidida em poucos dias",
    "Discussões sobre o novo governo e suas políticas",
    "Reunião sobre a nova legislação política e suas implicações",
    "Debate sobre as eleições e possíveis candidatos",
    "Análise das eleições e possíveis impactos na política",
    "Novas propostas de lei estão sendo discutidas no congresso",
    "A política internacional está em constante mudança",
    "Os principais candidatos discutiram suas plataformas em um debate",
    "A nova reforma tributária está em fase de aprovação",
    "Políticas ambientais estão sendo revisadas pelo governo",
    "O orçamento do governo para o próximo ano foi anunciado",
    "O impacto das políticas econômicas recentes está sendo avaliado",
    "A crise política em alguns países está afetando o comércio internacional"
]
categorias = [
    # Tecnologia
    "tecnologia", "tecnologia", "tecnologia", "tecnologia", "tecnologia", 
    "tecnologia", "tecnologia", "tecnologia", "tecnologia", "tecnologia",
    "tecnologia", "tecnologia", "tecnologia",
    
    # Esportes
    "esportes", "esportes", "esportes", "esportes", "esportes",
    "esportes", "esportes", "esportes", "esportes", "esportes",
    "esportes", "esportes", "esportes",
    
    # Política
    "política", "política", "política", "política", "política",
    "política", "política", "política", "política", "política",
    "política", "política", "política"
]


# Convertendo textos em uma matriz de contagens de tokens
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(textos)

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, categorias, test_size=0.5, random_state=42)

# Listando os classificadores para experimentar
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Naive Bayes": MultinomialNB(),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Perceptron": Perceptron()
}

# Treinando e avaliando cada classificador
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{name} - Accuracy: {accuracy_score(y_test, y_pred)}")