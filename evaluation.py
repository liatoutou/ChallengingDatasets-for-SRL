import json
from typing import List, Tuple
from allennlp.predictors.predictor import Predictor



def load_data(file_name: str) -> List[dict]:
    with open(file_name) as f:
        return json.load(f)


def generate_predictions(data: List[dict], model_predictor: Predictor) -> List[dict]:
    predicted_results = []

    for sentence_entry in data:
        input_sentence = ' '.join(sentence_entry["tokens"]).rstrip() + '.'
        prediction = model_predictor.predict(sentence=input_sentence)
        predicted_results.append(prediction)

    return predicted_results


def evaluate_predicted_arguments(predictions: List[dict], data: List[dict]) -> Tuple[int, int, int]:
    total_sentences = len(predictions)
    successful_sentences = 0
    failed_sentences = 0

    for i in range(len(predictions)):
        current_prediction = predictions[i]
        current_sentence = data[i]
        is_arg0_correct = False
        is_arg1_correct = False

        if "ARG0" in current_sentence["BIO"]:
            arg0_token = current_sentence["tokens"][current_sentence["BIO"].index("ARG0")]

            for verb in current_prediction["verbs"]:
                if arg0_token.lower() in verb["description"].lower():
                    is_arg0_correct = True
                    break

        if "ARG1" in current_sentence["BIO"]:
            arg1_token = current_sentence["tokens"][current_sentence["BIO"].index("ARG1")]

            for verb in current_prediction["verbs"]:
                if arg1_token.lower() in verb["description"].lower():
                    is_arg1_correct = True
                    break

        if not (is_arg0_correct and is_arg1_correct):
            failed_sentences += 1
        else:
            successful_sentences += 1

    return successful_sentences, total_sentences, failed_sentences


def main():
    test_files = ["active_passive.json", "clefts.json", "statement_question.json", "ellipsis.json","polysemy.json","conative.json","synonymy.json"]
    for m in range(2):
        if m == 0:
            model_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/structured-prediction-srl-bert.2020.12.15.tar.gz")
            print("Now evaluating model: structured-prediction-srl-bert ")
            for test_file in test_files:
                data = load_data(test_file)
                predictions = generate_predictions(data, model_predictor)
                successful_sentences, total_sentences, failed_sentences = evaluate_predicted_arguments(predictions, data)

                print(f"Performance on {test_file}")
                print(f"{successful_sentences} passes out of {total_sentences}")
                failure_percentage = round((failed_sentences / total_sentences) * 100, 2)
                print(f"{failure_percentage}% failures")
                print("----------------------------------")
                with open(f"outputs/{m}_{test_file}", "w") as f:
                    json.dump(predictions, f)
        if m == 1:
            model_predictor = Predictor.from_path(
                "https://storage.googleapis.com/allennlp-public-models/openie-model.2020.03.26.tar.gz")
            print("Now evaluating model: structured-prediction-srl")
            for test_file in test_files:
                data = load_data(test_file)
                predictions = generate_predictions(data, model_predictor)
                successful_sentences, total_sentences, failed_sentences = evaluate_predicted_arguments(predictions, data)

                print(f"Performance on {test_file}")
                print(f"{successful_sentences} passes out of {total_sentences}")
                failure_percentage = round((failed_sentences / total_sentences) * 100, 2)
                print(f"{failure_percentage}% failures")
                print("----------------------------------")
                with open(f"outputs/{m}_{test_file}", "w") as f:
                    json.dump(predictions, f)


if __name__ == "__main__":
    main()