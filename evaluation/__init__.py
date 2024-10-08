from .bleu import Bleu
from .rouge import Rouge
from .cider import Cider
from .accuracy import Accuracy
from .f1 import F1

def compute_scores(gts, gen):
    metrics = (F1(), Accuracy(), Cider(), Rouge(), Bleu())
    all_score = {}
    all_scores = {}
    for metric in metrics:
        score, scores = metric.compute_score(gts, gen)
        all_score[str(metric)] = score
        all_scores[str(metric)] = scores

    return all_score, all_scores