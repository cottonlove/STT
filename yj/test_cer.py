import pandas as pd
import os
from modules.vocab import KoreanSpeechVocabulary
import Levenshtein as Lev

def get_metric(metric_name, vocab):
    if metric_name == 'CER':
        return CharacterErrorRate(vocab)
    else:
        raise ValueError('Unsupported metric: {0}'.format(metric_name))


def distance(s1, s2):
    if not s1:
        return len(s2)

    VP = (1 << len(s1)) - 1
    VN = 0
    currDist = len(s1)
    mask = 1 << (len(s1) - 1)

    block = {}
    block_get = block.get
    x = 1
    for ch1 in s1:
        block[ch1] = block_get(ch1, 0) | x
        x <<= 1

    for ch2 in s2:
        # Step 1: Computing D0
        PM_j = block_get(ch2, 0)
        X = PM_j
        D0 = (((X & VP) + VP) ^ VP) | X | VN
        # Step 2: Computing HP and HN
        HP = VN | ~(D0 | VP)
        HN = D0 & VP
        # Step 3: Computing the value D[m,j]
        currDist += (HP & mask) != 0
        currDist -= (HN & mask) != 0
        # Step 4: Computing Vp and VN
        HP = (HP << 1) | 1
        HN = HN << 1
        VP = HN | ~(D0 | HP)
        VN = HP & D0

    return currDist


class ErrorRate(object):
    """
    Provides inteface of error rate calcuation.

    Note:
        Do not use this class directly, use one of the sub classes.
    """

    def __init__(self, vocab) -> None:
        self.total_dist = 0.0
        self.total_length = 0.0
        self.vocab = vocab

    def __call__(self, targets, y_hats):
        """ Calculating character error rate """
        dist, length = self._get_distance(targets, y_hats)
        self.total_dist += dist
        self.total_length += length
        return self.total_dist / self.total_length

    def _get_distance(self, targets, y_hats):
        """
        Provides total character distance between targets & y_hats

        Args:
            targets (torch.Tensor): set of ground truth
            y_hats (torch.Tensor): predicted y values (y_hat) by the model

        Returns: total_dist, total_length
            - **total_dist**: total distance between targets & y_hats
            - **total_length**: total length of targets sequence
        """
        total_dist = 0
        total_length = 0

        for (target, y_hat) in zip(targets, y_hats):
            s1 = self.vocab.label_to_string(target)
            s2 = self.vocab.label_to_string(y_hat)

            dist, length = self.metric(s1, s2)

            total_dist += dist
            total_length += length

        return total_dist, total_length

    def metric(self, *args, **kwargs):
        raise NotImplementedError


class CharacterErrorRate(ErrorRate):
    """
    Computes the Character Error Rate, defined as the edit distance between the
    two provided sentences after tokenizing to characters.
    """
    def __init__(self, vocab):
        super(CharacterErrorRate, self).__init__(vocab)

    def metric(self, s1: str, s2: str):
        """
        Computes the Character Error Rate, defined as the edit distance between the
        two provided sentences after tokenizing to characters.

        Arguments:
            s1 (string): space-separated sentence
            s2 (string): space-separated sentence
        """
        s1 = s1.replace(' ', '')
        s2 = s2.replace(' ', '')

        # if '_' in sentence, means subword-unit, delete '_'
        if '_' in s1:
            s1 = s1.replace('_', '')

        if '_' in s2:
            s2 = s2.replace('_', '')

        dist = distance(s2, s1)
        length = len(s1.replace(' ', ''))

        return dist, length
    
def cer(ref, hyp):
    ref = ref.replace(' ', '')
    hyp = hyp.replace(' ', '')
    dist = Lev.distance(hyp, ref)
    length = len(ref)
    return dist, length, dist/length


if __name__ == '__main__':
    #각 문장의 wer, cer은 inferenence_result 폴더의 csv 파일에 두 컬럼으로 저장하고
    #각 파일마다의 전체 wer, cer은 print하기
    inference_file = "ds2_encoder5_inference2.csv"
    GT_file = "test_transcript.csv"
    print("start")
    inference_data = pd.read_csv(inference_file)
    GT_data = pd.read_csv(GT_file)

    total_cer_score = 0
    print(inference_data)
    # Sort dataframes by the 'filename' column
    inference_data = inference_data.sort_values(by='audio_path')
    GT_data = GT_data.sort_values(by='audio_path')
    
    print(inference_data.iloc[0])
    print(GT_data.iloc[0])
    
#     vocab = KoreanSpeechVocabulary(os.path.join(os.getcwd(), 'labels.csv'), output_unit='character')
#     metric = get_metric(metric_name='CER', vocab=vocab)
    
    for i in range(len(inference_data)):
        ref = inference_data.iloc[i]['Inference_result']
        hyp = GT_data.iloc[i]['Korean_transcript']
        cer_score = cer(ref, hyp)[2]
        total_cer_score += cer_score
    print("ds2 encoder5  cer: ",total_cer_score/len(inference_data))
