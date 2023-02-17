def top_k(self, score, top_k):
    """
    From https://github.com/abhinanda-punnakkal/BABEL/blob/main/action_recognition/feeders/feeder.py
    """
    rank = score.argsort()
    hit_top_k = [l in rank[i, -top_k:] for i, l in enumerate(self.label[0])]
    return sum(hit_top_k) * 1.0 / len(hit_top_k)