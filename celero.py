import os
import random
import spacy
from spacy.util import minibatch, compounding
from spacy.training import Example

def load_training_data(
    data_directory: str = "aclImdb/train",
    limit: int = 0
) -> tuple:
    # Carregar arquivos
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label
                            }
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    return reviews

def load_test_data(
    data_directory: str = "aclImdb/test",
    limit: int = 0
) -> tuple:
    # Carregar arquivos
    reviews = []
    for label in ["pos", "neg"]:
        labeled_directory = f"{data_directory}/{label}"
        for review in os.listdir(labeled_directory):
            if review.endswith(".txt"):
                with open(f"{labeled_directory}/{review}") as f:
                    text = f.read()
                    text = text.replace("<br />", "\n\n")
                    if text.strip():
                        spacy_label = {
                            "cats": {
                                "pos": "pos" == label,
                                "neg": "neg" == label
                            }
                        }
                        reviews.append((text, spacy_label))
    random.shuffle(reviews)

    if limit:
        reviews = reviews[:limit]
    return reviews

def evaluate_model(
    tokenizer, textcat, test_data: list
) -> dict:
    reviews, labels = zip(*test_data)
    reviews = (tokenizer(review) for review in reviews)
    true_positives = 0
    false_positives = 1e-8  # Não pode ser 0 pela presença no denominador
    true_negatives = 0
    false_negatives = 1e-8
    for i, review in enumerate(textcat.pipe(reviews)):
        true_label = labels[i]
        for predicted_label, score in review.cats.items():
            # Todo dicionário inclui ambas as labels. Pode-se ter toda a informação necessária
            # apenas com a label "pos".
            if (
                predicted_label == "neg"
            ):
                continue
            if score >= 0.5 and true_label[list(true_label)[0]]["pos"]:
                true_positives += 1
            elif score >= 0.5 and true_label[list(true_label)[0]]["neg"]:
                false_positives += 1
            elif score < 0.5 and true_label[list(true_label)[0]]["neg"]:
                true_negatives += 1
            elif score < 0.5 and true_label[list(true_label)[0]]["pos"]:
                false_negatives += 1
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)

    if precision + recall == 0:
        f_score = 0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precisao": precision, "recall": recall, "f-score": f_score}

def train_model(
    training_data: list,
    test_data: list,
    iterations: int = 20
) -> None:
    # Construir o pipeline
    nlp = spacy.load("en_core_web_sm")
    if "textcat" not in nlp.pipe_names:
        textcat = nlp.add_pipe("textcat")
    else:
        textcat = nlp.get_pipe("textcat")

    textcat.add_label("pos")
    textcat.add_label("neg")

    # Treinar apenas textcat
    training_excluded_pipes = [
        pipe for pipe in nlp.pipe_names if pipe != "textcat"
    ]
    with nlp.disable_pipes(training_excluded_pipes):
        optimizer = nlp.begin_training()
        # Loop de treinamento
        print("Iniciando treinamento")
        print("Loss\tPrecisao\tRecall\tF-score")
        batch_sizes = compounding(
            4.0, 32.0, 1.001
        )  # Um gerador que produz uma série infinita de números de entrada
        for i in range(iterations):
            print(f"Iteracao de treinamento {i}")
            loss = {}
            random.shuffle(training_data)
            batches = minibatch(training_data, size=batch_sizes)
            for batch in batches:
                examples = []
                for text, annots in batch:
                    examples.append(Example.from_dict(nlp.make_doc(text), annots))
                nlp.update(examples, losses=loss)
            with textcat.model.use_params(optimizer.averages):
                evaluation_results = evaluate_model(
                    tokenizer=nlp.tokenizer,
                    textcat=textcat,
                    test_data=test_data
                )
                print(
                    f"{loss['textcat']}\t{evaluation_results['precision']}"
                    f"\t{evaluation_results['recall']}"
                    f"\t{evaluation_results['f-score']}"
                )

    # Salvar o modelo
    with nlp.use_params(optimizer.averages):
        nlp.to_disk("model_artifacts")



#def test_model(input_data: str = TEST_REVIEW):
def test_model():
    file =  input('Insira o caminho para o arquivo com o texto a ser revisto: ')
    f = open(file)
    input_data = f.read()
    input_data = input_data.replace("<br />", "\n\n")
    #  Carregar o modelo treinado salvo
    loaded_model = spacy.load("model_artifacts")
    # Gerar predição
    parsed_text = loaded_model(input_data)
    # Determinar a predição a ser retornada
    if parsed_text.cats["pos"] > parsed_text.cats["neg"]:
        prediction = "(1) Positivo"
        score = parsed_text.cats["pos"]
    else:
        prediction = "(-1) Negativo"
        score = parsed_text.cats["neg"]
    print(
        f"Texto da review: {input_data}\nSentimento: {prediction}"
        f"\tScore: {score}"
    )

if __name__ == "__main__":
    train = load_training_data()
    test = load_test_data()
    train_model(train, test)
    print("Testando modelo")
    test_model()