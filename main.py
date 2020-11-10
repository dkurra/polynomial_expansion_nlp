import sys
from typing import Tuple

MAX_SEQUENCE_LENGTH = 29
TRAIN_URL = ""

from model import EquationSolver
from utils import *
import pickle
import itertools



def load_file(file_path: str) -> Tuple[Tuple[str], Tuple[str]]:
    """ A helper functions that loads the file into a tuple of strings

    :param file_path: path to the data file
    :return factors: (LHS) inputs to the model
            expansions: (RHS) group truth
    """
    data = open(file_path, "r").readlines()
    factors, expansions = zip(*[line.strip().split("=") for line in data])
    return factors, expansions


def score(true_expansion: str, pred_expansion: str) -> int:
    """ the scoring function - this is how the model will be evaluated

    :param true_expansion: group truth string
    :param pred_expansion: predicted string
    :return:
    """
    return int(true_expansion == pred_expansion)


# --------- START OF IMPLEMENT THIS --------- #
def predict(factors: str):
    """
         predicts inputs in batches of 32 or single input
        :param factors: list of strings (factors) or string
        :return: list of strings (expansions) or expansion for input
    """
    if type(factors) is not list:
        factors = [factors]

    preds = []
    # model has two inputs (encoder and decoder)
    X_new, X_decoder = prepare_new_sequences(factors, tokenizer)
    Y_probas = model.predict([X_new, X_decoder])

    Y_pred = tf.argmax(Y_probas, axis=-1)
    preds.append(tokenizer.sequences_to_texts(Y_pred.numpy().tolist()))
    return list(map(process_result, itertools.chain(*preds)))[0]


predict_single = predict


def main(filepath: str):
    factors, expansions = load_file(filepath)
    pred = [predict(f) for f in factors]
    scores = [score(te, pe) for te, pe in zip(expansions, pred)]
    print(np.mean(scores))

def predict_batch(factors: str):
    """
     predicts inputs in batches of 32
    :param factors: list of strings (factors)
    :return: list of strings (expansion)
    """
    factors = list(factors)
    preds = []
    for e in chunks(factors, 32):
        X_new, X_decoder = prepare_new_sequences(e, tokenizer)
        Y_probas = model.predict([X_new, X_decoder])

        Y_pred = tf.argmax(Y_probas, axis=-1)
        preds.append(tokenizer.sequences_to_texts(Y_pred.numpy().tolist()))
    return list(map(process_result, itertools.chain(*preds)))


def main_batch(filepath: str):
    factors, expansions = load_file(filepath)
    preds = predict_batch(factors)
    scores = [score(te, pe) for te, pe in zip(expansions[0:100], preds)]
    print(np.mean(scores))



def load_model():
    """
        loads a saved model. for more details refer expand.ipynb
        :return: keras.models.Model
    """
    global model
    model = EquationSolver()
    model.load_weights('saved_model/equ_model_2')


def load_tokenizer():
    """
        loads a saved tokenizer. for more details refer expand.ipynb
        :return: keras_preprocessing.text.Tokenizer
    """
    global tokenizer
    with open('saved_model/tokenizer.pickle', 'rb') as f:
        tokenizer = pickle.load(f)


tokenizer = None
model = None

# --------- END OF IMPLEMENT THIS --------- #


if __name__ == "__main__":
    """
        please refer to expand.ipynb jupyter notebook for detailed explanation of strategy, data preparation, 
        modeling and, evaluation
        
        Note: input factors need to be a subset of : ['*', '-', '2', '(', ')', '1', '+', '=', '4', 
        '3', '6', '5', '8', '7', '0', 's', 'n', 'i', '9', 't', 'a', 'c', 'o', 'y', 'z', 'k', 'h', 'j', 'x', 'b', 'd', 
        'e', 'f', 'g', 'l', 'm', 'p', 'q', 'r', 'u', 'v', 'w']
    """
    load_tokenizer()
    load_model()
    main("test.txt" if "-t" in sys.argv else "train.txt")
    # uncomment below line if you want to inference in batches of 32
    # main_batch("test.txt" if "-t" in sys.argv else "train.txt")
